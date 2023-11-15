from transformers import AdamW, AutoConfig
from sentence_transformers import models, SentenceTransformer, losses
import torch

import logging

logger = logging.getLogger(__name__)

# Python
import ssl  # Get sure you can download punk from nltk

# Modules
from src.data_augmentation import get_maximums, score_balancer, resize_join_synthetic, resize_join_multiplying
from src.pretraining_tools import scoring, sbert_dataloader, encoding_classification, combinations, \
    train_test_spliter, evaluator_
from src.tools import check_device, read_csv

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# nltk.download("punkt")


def trainset_sbert(df, model_type, BATCH, scoring_ratio=1, *weights):
    """ This function creates samples for contrastive learning based on a score.
        - weights (floats):  are the weights per level.
        - df (pd.DF): df is the dataframe with the training samples and it must contain sentence1, sentence2, cat1_1
            cat1_2, cat2_1 and cat2_2 as columns
    """
    df = scoring(df, model_type, *weights)
    if "class" in model_type:
        counts = df["score"].value_counts()
        threshold = counts.min() * scoring_ratio
        df = score_balancer(df, counts, threshold, "Training")
    else:
        pass
    train_dataloader = sbert_dataloader(df, BATCH)
    return train_dataloader


def sbert_data_preproces(sbert_data, resize_data_syn, resize_data_proportion, dataset_size, evaluation_metric,
                         level, model_type, train_portion, eval_data, scoring_ratio, ratio_max,
                         classic, BATCH, Augmentation, ratio_min, *weights):
    """ This function preprocess the data for the phase 2.
        1. resize the dataset
        2.
    """
    # Initialization
    logger.debug("Starting DATA preprocess ...\n")
    sbert_data = read_csv(sbert_data) if isinstance(sbert_data, str) else sbert_data
    split = 1 if eval_data else train_portion
    # The lowest one in the hierarchical structure
    deepest_cat = [x for x in sbert_data.columns.to_list() if "cat" in x][0]

    # ######### Balance in the dataset
    logger.debug("No labels from level {}: {}".format(level, len(list(sbert_data[deepest_cat].unique()))))
    if "class" in model_type:
        logger.debug("Counts for level {}: {}".format(level, sbert_data[deepest_cat].value_counts()))
    logger.debug("Training dataset has shape {} \n".format(sbert_data.shape))

    if resize_data_syn:
        # Synthetic
        maxn, df_list = get_maximums(sbert_data, level, 0)
        resized = resize_join_synthetic(df_list, maxn, classic, ratio_min=ratio_min, ratio_max=ratio_max)
        logger.debug("Training data after resize_join_synthetic has shape {} \n".format(resized.shape))
    else:
        logger.debug(" No resize_join_synthetic has happened")

    if resize_data_proportion:
        # Multiplying
        resized = resized if resize_data_syn else sbert_data
        maxn, df_list = get_maximums(resized, level, 0)
        sbert_data = resize_join_multiplying(df_list, maxn, classic, ratio_min=ratio_min, ratio_max=ratio_max)
        logger.debug("Training data after resize_join_multiplying has shape {} \n".format(sbert_data.shape))
    else:
        logger.debug(" No resize_join_multiplying has happened")

    # ######### ENCODING
    if "class" in model_type:
        sbert_data = encoding_classification(sbert_data, hierarchy_level=level)

    # COMBINATIONS
    evaluation3 = combinations(sbert_data, dataset_size=dataset_size, multiply_by=Augmentation, level=level)
    train, test = train_test_spliter(evaluation3, split, eval_data)
    train_dataloader = trainset_sbert(train, model_type, BATCH, scoring_ratio, *weights)
    if split == 1:
        test = read_csv(eval_data)
        test = combinations(test, dataset_size="extra_big", multiply_by=Augmentation, level=level)
    evaluator = evaluator_(test, evaluation_metric, model_type, *weights)
    return train_dataloader, evaluator


def training_sbert(model_name,
                   evaluation_metric,
                   level,
                   model_type,
                   model_path="",
                   dataset_size="medium",
                   base_type="stransformer",
                   resize_data_syn=False,
                   resize_data_proportion=False,
                   data_path=None,
                   eval_data=None,
                   scoring_ratio=1,
                   ratio_max=8,
                   classic=True,
                   pooling_mode='cls',
                   train_portion=0.9,
                   epochs=None,
                   LR=None,
                   Augmentation=None,
                   BATCH=0,
                   ratio_min=1.5, *weights
                   ):
    """
    This is the wrapper for the second phase. One needs to pay attention to the parameters regarding to the
    data augmentation and the pair creation.
    We can create millions of samples by combinatorics, so we restrict to some amount determined by multi,
    this is the number of times that each sentence appears in the combinations.
    The data augmentation is to balance the dataset by oversampling. There are two approaches implemented:
    synthetic, by considering those queries from underrepresented classes as well as themselves without stopwords,
    and data augmentation by duplication of data. The second one makes sense in our case since the probability of
    having a duplicated is almost zero. By default, the proportion of occurrences fo the queries from underrepresented
    classes will never be bigger than 20 times those from the biggest class if classic = False.
    Something similar is done with the number of score values with scoring_ratio.

    Input:
    - pooling_mode: pooling mode
    - model_name: str, name given to the model
    - evaluation_metric: str, either binary, considering equal or not, or sim that evaluates based on the levels
    - level: int, number of degrees in the hierarchy
    - model_type: str, "classification" or "regression"
    - model_path: path to the model from phase 1 that is going to be further trained
    - dataset_size: str, small, medium, big or extra_big (0.1, 0.25, 0.5, 1).
        It is the ratio of samples taken after the creation of the pairs.
    - base_type: str, "stransformer" or "transformer". The output is going to be a stransformer so this refers
        to the model from model_path.
    - resize_data_syn: Boolean, if yes data augmentation deleting stopwords is applied
    - resize_data_proportion: Boolean, if yes data augmentation deleting stopwords is applied
    - data_path: path, path to the data used for training
    - eval_data: path, path to the data used for validation
    - scoring_ratio: Proportion between scoring values, only works for classification (check trainset_sbert)
    - ratio_max: int, max number of times that each sentence appears in the combinations
    - classic: Boolean, how to resize underrepresented classes. False, then it is given by a polynomial
        function with no more than 20 times ratio.
    - weights: floats, weights per level. They must match the number of levels
    - train_portion: int, ratio of training data
    Output:
    - path, where the models has been saved
    """
    # Preprocessing
    torch.cuda.empty_cache()
    device = check_device()
    sbert_data = read_csv(data_path)
    sbert_data = sbert_data.dropna(how="any")
    train_dataloader, evaluator = sbert_data_preproces(
        sbert_data,
        resize_data_syn,
        resize_data_proportion,
        dataset_size,
        evaluation_metric,
        level,
        model_type,
        train_portion,
        eval_data,
        scoring_ratio,
        ratio_max,
        classic,
        BATCH,
        Augmentation,
        ratio_min,
        *weights
    )
    # Model creation
    if base_type == "stransformer":
        model = SentenceTransformer(model_path, device=device)
        train_loss = losses.CosineSimilarityLoss(model)
        logger.debug("The base model is a sentence-transformer\n")
    else:
        config = AutoConfig.from_pretrained(model_path)
        max_length = config.max_position_embeddings - 2
        word_embedding_model = models.Transformer(model_path,
                                                  max_seq_length=max_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode=pooling_mode)
        model = SentenceTransformer(modules=[word_embedding_model,
                                             pooling_model],
                                    device=device)
        logger.debug("The base model is a transformers\n")
        train_loss = losses.CosineSimilarityLoss(model)

    # Hyperparameters
    steps_per_epoch = len(train_dataloader)
    final_2 = model_name

    # Tune the model
    logger.debug("\n Start training phase 2 ... \n")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        scheduler="warmuplinear",
        warmup_steps=int((len(train_dataloader) * epochs) / 3),
        optimizer_class=AdamW,
        optimizer_params={"lr": LR},
        weight_decay=0,
        evaluation_steps=int((steps_per_epoch * epochs) / 10),
        output_path=final_2,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True
    )
    logger.debug("Hecho!")

    return final_2
