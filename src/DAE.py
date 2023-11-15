# Python modules
import ssl
from typing import List

# Libraries
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, datasets, losses

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import torch

# Make sure you can download punk from nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import numpy as np

from src.pretraining_tools import evaluator_, encoding_classification
from src.tools import get_base, check_device

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt')
import logging

logger = logging.getLogger(__name__)


class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """
    def __init__(self, sentences: List[str], del_ratio=0.6):
        self.sentences = sentences
        self.noise_fn = lambda s: DenoisingAutoEncoderDataset.delete(s, del_ratio=del_ratio)
        print(f"\n, noise_fn:{self.noise_fn} ,\n")

    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed


def max_length_checker_transformer(list_queries, tokenizer=None, encoder=None):
    """ This function computes the max length for the model with a 20% of range """
    # Tokenizer
    try:
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(encoder, do_lower_case=True)
    except:
        logger.debug("There was a problem getting the max length from the model we are going to use {}".format(100))
        return 100

    # Max length for the model based on the
    max_length = 0
    print("\n Computing the max length for our models input based on the samples")
    list_queries = [v for v in list_queries if len(v) != 0]
    for sent in tqdm(list_queries):
        new_len = len(tokenizer.encode(sent))
        if new_len > max_length:
            max_length = new_len
    max_position_embeddings = int(AutoConfig.from_pretrained(encoder).max_position_embeddings - 2)
    max_length = int(max_length * 1.2)
    max_length = min([max_length, max_position_embeddings])
    print("\n max length:{}".format(max_length))
    return max_length


def max_length_checker_stransformer(list_queries, model, encoder=None):
    """ This function computes the max length for the model with a 20% of range """
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder)
    except:
        logger.debug("There was a problem getting the max length from the model we are going to use {}".format(100))
        return model.max_seq_length

    # Max length for the model based on the
    max_length = 0
    print("\n Computing the max length for our models input based on the samples")
    list_queries = [v for v in list_queries if len(v) != 0]
    for sent in tqdm(list_queries):
        new_len = len(tokenizer.encode(sent))
        if new_len > max_length:
            max_length = new_len
    max_length = int(max_length * 1.2)
    max_length = min([max_length, model.max_seq_length])
    print("\n max length:{}".format(max_length))
    return max_length


def max_len_wrapper(list_queries, use_max_len, max_len, encoder_path, base_type, device):
    """ This function defines the max length for the model and assemble the pieces. """
    # Remark: the max length by default for all the sentence embedder model
    # from the library sentence bert is 128 tokens. This must be changed !
    # This model admits different transformer and stransformer models.
    if base_type == "transformer":
        if max_len == 0 and use_max_len:
            max_len = max_length_checker_transformer(list_queries, encoder=encoder_path)
        elif not use_max_len:
            max_len = AutoConfig.from_pretrained(encoder_path).max_position_embeddings
        word_embedding_model = models.Transformer(encoder_path, max_seq_length=max_len)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode='cls')
        logger.debug("The model was successfully defined")
        # Join base model and head
        model = SentenceTransformer(modules=[word_embedding_model,
                                             pooling_model], device=device)
        logger.debug("We set the max length to {}".format(max_len))
        model.max_seq_length = max_len
    elif base_type == "stransformer":
        logger.debug("Encoder path{}".format(encoder_path))
        model = SentenceTransformer(encoder_path, device=device)
        if max_len == 0:
            max_len = max_length_checker_transformer(list_queries, model, encoder_path)
        print("We set the max length to {}".format(max_len))
        model.max_seq_length = max_len
    return model, max_len


def load_tsdae_data(train_path, val_path):
    logger.debug("\nLoading datasets: {} for training and {} for the evaluator".format(
        get_base(train_path), get_base(val_path)))

    # Loading training dataset
    try:
        dae_data = pd.read_csv(train_path, index_col=False, names=["name_query"], encoding='utf-16')
    except:
        dae_data = pd.read_csv(train_path, index_col=False, names=["name_query"], encoding='utf-8')
    logger.debug("Number of samples for training {}".format(dae_data.shape[0]))
    dae_data["name_query"] = dae_data["name_query"].apply(str)
    dae_data = dae_data.sample(frac=1)
    # Loading testing dataset
    try:
        val_data = pd.read_csv(val_path, index_col=False, encoding='utf-16')
    except:
        val_data = pd.read_csv(val_path, index_col=False, encoding='utf-8')
    val_data = val_data.sample(frac=1)
    return dae_data, val_data


def tsdae_train_dataset(list_queries, BATCH, del_ratio):
    """Create the special denoising DATASET that adds noise on-the-fly.
    This module deletes with a ratio =0.6 at the word level"""
    train_dataset = DenoisingAutoEncoderDataset(list_queries, del_ratio)
    return DataLoader(train_dataset, batch_size=BATCH, shuffle=True)


def tsdae_evaluator(df, evaluation_metric, model_type, level, *weights):
    # Encoding
    if "class" in model_type:
        logger.debug("Classification")
        df = encoding_classification(df, hierarchy_level=level)

    # Create the validation dataset by combinations
    evaluation1 = df.sample(frac=0.20).reset_index(drop=True)
    evaluation2 = df.sample(frac=0.20).reset_index(drop=True)
    evaluation1.columns = ["sentence1"] + ["cat1_" + str(i + 1) for i in range(level)]
    evaluation2.columns = ["sentence2"] + ["cat2_" + str(i + 1) for i in range(level)]

    # Evaluator
    evaluation3 = pd.concat([evaluation1, evaluation2], axis=1)
    return evaluator_(evaluation3, evaluation_metric, model_type, *weights)


def dae_trainer(model_name,
                data_path,
                val_path,
                evaluate,
                model_type,
                evaluation_metric="sim",
                use_max_len=False,
                max_len=0,
                base_type="transformer",
                encoder=None,
                level=2,
                VAE=False,
                epochs=1,
                BATCH=0,
                LR=5e-05,
                del_ratio=0.6):
    torch.cuda.empty_cache()

    # Initialize
    device = check_device()
    print(f"Device for phase 1{device}")
    logger.debug("\n\n DEVICE {} \n\n".format(device))
    weights = [1 / level for _ in range(level)]

    # Data sets: train just a series with text and val_data with labels
    data_dae, val_data = load_tsdae_data(train_path=data_path, val_path=val_path)
    data_dae = pd.concat([data_dae] * epochs, axis=0, ignore_index=True)

    # Evaluator
    if evaluate:
        logger.debug(evaluation_metric)
        evaluator = tsdae_evaluator(val_data,
                                    evaluation_metric,
                                    model_type,
                                    level,
                                    *weights)
    else:
        evaluator = None

    # Dataloader
    list_queries = data_dae["name_query"].to_list()
    print("list_queries", list_queries[:2])
    train_dataloader = tsdae_train_dataset(list_queries, BATCH, del_ratio)

    # Model
    print(f"\n\nencoder_path {encoder}")
    print(f"MAX_LEN {max_len}")
    print(f"base_type {base_type}")
    print(f"use_max_len {use_max_len}\n\n")
    model, max_len = max_len_wrapper(list_queries[::2], use_max_len,
                                     max_len, encoder, base_type, device)

    # It has a cross entropy loss in the back
    if VAE:
        train_loss = losses.VAELoss(model,
                                    latent_dim=256,
                                    decoder_name_or_path=encoder,
                                    tie_encoder_decoder=True)
        logger.debug("VAE")
    else:
        train_loss = losses.DenoisingAutoEncoderLoss(model,
                                                     decoder_name_or_path=encoder,
                                                     tie_encoder_decoder=True)

    steps_per_epoch = len(train_dataloader)
    final = model_name
    evaluation_steps = int(steps_per_epoch / 5) if evaluate else 0
    # int((steps_per_epoch * epochs) / 5) if evaluate else 0
    # print("steps_per_epoch", steps_per_epoch)
    # print("evaluation_steps", evaluation_steps)
    # print("BATCH", BATCH)

    # Training
    logger.debug("The maximum length accepted by the model is: {}".format(model.max_seq_length))
    logger.debug("\n Start training Phase 1...")
    logger.debug("The maximum length accepted by the model is: {}".format(model.get_max_seq_length()))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,  # epochs
        scheduler="warmuplinear",
        warmup_steps=int(steps_per_epoch / 2),  # int((steps_per_epoch * epochs) / 2),
        optimizer_class=AdamW,
        optimizer_params={"lr": LR},
        weight_decay=0,
        evaluation_steps=evaluation_steps,
        output_path=final,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True
    )
    return final
