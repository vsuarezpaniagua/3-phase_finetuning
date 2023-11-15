import pickle
from os.path import join

import numpy as np
import pandas as pd
import torch
import os
import re

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from transformers import AdamW, get_linear_schedule_with_warmup

from src.config import RESULTS, PHASE1_TRAIN, PHASE2_TRAIN, PHASE3_TRAIN, PHASE3_ONLY_TRAIN, MULTI_PRETRAIN
from src.trainers import Dataset_from_sentences, validation, training

import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


#   ------------------------------------------------  GENERAL  ------------------------------------------------

def get_base(path):
    """ gets the name of the model from the paths of the model"""
    return os.path.basename(path).replace("final_", "")


def check_device(printit=False):
    """ This function checks the cuda's setting. It prints the setting. """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if printit:
            logger.debug('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.debug('We will use the GPU:', torch.cuda.get_device_name(0))
            logger.debug("Cuda's architectures is {}".format(torch.cuda.get_arch_list()))
            logger.debug("Device capability {}".format(torch.cuda.get_device_capability()))
    else:
        if printit:
            logger.debug('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def read_csv(path):
    print(f"path {path}")
    try:
        df = pd.read_csv(path, index_col=False, encoding='utf-16')
    except:
        df = pd.read_csv(path, index_col=False, encoding='utf-8')
    return df


def spliter(df, category_col_name, split=0.8):
    """ This function splits 80/20 by default a dataframe creating a new column data_type
    with the split "train"/"val" """
    # --- Random split based on index and labels
    if split == 1:
        df['data_type'] = ['train'] * df.shape[0]
    if split == 0:
        df['data_type'] = ['val'] * df.shape[0]
    else:
        X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                          df[category_col_name].values,
                                                          test_size=1 - split,
                                                          random_state=42,
                                                          stratify=df[category_col_name].values)

        # --- Data type columns for train and val
        df['data_type'] = ['not_set'] * df.shape[0]
        df.loc[X_train, 'data_type'] = 'train'
        df.loc[X_val, 'data_type'] = 'val'
    return df


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def mse_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return mean_squared_error(labels_flat, preds_flat)


def check_dir(path):
    """ For a given file does may not exit it checks if the parent directory exists and
        if it does not it is created. """
    path = Path(path)
    dirName = path.parent.absolute()
    try:
        os.makedirs(dirName)
        logger.debug("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


#   ------------------------------------------------  TRAINING  ------------------------------------------------


def text_split(text, tokenizer, max_len):
    """This function gets the number of tokens in an input and if it is longer than
    the maximum then it splits into sentences as long as possible"""
    try:
        my_len = len(tokenizer(text)["input_ids"])
    except:
        print(f"###########{text}")
    if my_len > max_len:
        output = []
        sub_text = text.split(".")
        total_text, current_text = "", ""
        for t in sub_text:
            t += "."
            current_text += t
            if len(tokenizer(current_text)["input_ids"]) < max_len:
                total_text = current_text
            else:
                output.append(total_text)
                current_text = t
        current_text = current_text[:-1]
        output.append(current_text)
    else:
        output = [text]
    return output


def chunk_it(df2, tokenizer, max_len):
    """ This function explodes the input text examples. """
    df2["query"] = df2["query"].apply(lambda x: text_split(x, tokenizer, max_len))
    df2 = df2.explode('query')
    return df2["query"], df2


def create_val_test_train(test_data, train_data_original, model_type, BATCH):
    """ This function prepares the data for training the last phase, the fine-tuning.
        We us ethe test_data if it is not None. Otherwise, we extract the 20% of the train set.
    """
    if not test_data or test_data is None:
        X_train, X_test, y_train, y_test = train_test_split(train_data_original["query"].to_list(),
                                                            train_data_original["cat1"].to_list(),
                                                            test_size=0.2, random_state=42)

        train_data1 = Dataset_from_sentences(X_train, model_type, y_train)
        val_data1 = Dataset_from_sentences(X_test, model_type, y_test)

        train_dataloader = DataLoader(train_data1,
                                      shuffle=True,
                                      # sampler=RandomSampler(train_data1),
                                      batch_size=BATCH)
        test_dataloader = DataLoader(val_data1,
                                     shuffle=True,
                                     batch_size=BATCH)
    else:
        train_data = Dataset_from_sentences(train_data_original["query"].to_list(), model_type,
                                            train_data_original["cat1"].to_list())

        test_data = pd.read_csv(test_data)
        test_data = test_data.dropna(how="any")
        test_data = Dataset_from_sentences(test_data["query"].to_list(), model_type,
                                           test_data["cat1"].to_list())

        train_dataloader = DataLoader(train_data,
                                      shuffle=True,
                                      # sampler=RandomSampler(train_data),
                                      batch_size=BATCH)
        test_dataloader = DataLoader(test_data, shuffle=True, batch_size=BATCH)

    return train_dataloader, test_dataloader


def trainer(EPOCHS_MIN, EPOCHS_MAX, train_dataloader, test_dataloader, model,
            optimizer, device, scheduler, model_type, average):
    metric_dict = dict(zip(["mse", "ACC", "P", "R", "F1", 'confusion'],
                           [list(), list(), list(), list(), list(), list()]))
    last_val_loss, test_loss = 1001, 1000
    earlystop, counter, restart = 0, 0, 0
    restart_max = min(EPOCHS_MAX / 2, EPOCHS_MIN * 2)
    while earlystop < EPOCHS_MIN and counter < EPOCHS_MAX and restart < restart_max:
        counter += 1
        if last_val_loss < test_loss:
            earlystop += 1
        elif earlystop != 0:
            restart += 1
            earlystop = 0
        last_val_loss = test_loss
        if "reg" in model_type:
            # Training
            train_loss, referece_metric = training(train_dataloader,
                                                   model,
                                                   optimizer,
                                                   device,
                                                   scheduler,
                                                   model_type)
            # Validation
            test_loss, mse = validation(test_dataloader, model, device, model_type, average)
            print("### test_loss: {:.4}\n  mse: {:.2}\n".format(test_loss, mse))
            metric_dict["mse"].append(mse)
        if "class" in model_type:
            # Training
            train_loss, m1, _ = training(train_dataloader,
                                         model,
                                         optimizer,
                                         device,
                                         scheduler,
                                         model_type)
            print(f"### train_loss: {train_loss:.4} ACC: {m1[0]:.4}\tP: {m1[1]:.4}\tR: {m1[2]:.4}\tF1: {m1[3]:.4}")
            # Validation
            test_loss, m, confusion = validation(test_dataloader, model, device, model_type, average)
            print(f"### test_loss:   {test_loss:.4} ACC: {m[0]:.4}\tP: {m[1]:.4}\tR: {m[2]:.4}\tF1: {m[3]:.4}")
            metric_dict["ACC"].append(m[0])
            metric_dict["P"].append(m[1])
            metric_dict["R"].append(m[2])
            metric_dict["F1"].append(m[3])
            metric_dict["confusion"].append(confusion)
    return train_loss, metric_dict


def optimization_setup(train_dataloader, model, LR, EPSILON, EPOCHS):
    steps_per_epoch = len(train_dataloader)
    optimizer = AdamW(
        model.parameters(),
        lr=LR,  # 2e-5 based on original bert paper
        eps=EPSILON  # this is default
    )
    sch_rate = steps_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * sch_rate),
        num_training_steps=sch_rate
    )
    return scheduler, optimizer


def save_and_plot(mydict, mystring, model_name):
    mystring = mystring + "_" + model_name
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)
    with open(join(RESULTS, mystring + '.pkl'), 'wb') as handle:
        pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Show/save figure as desired.
    plt.figure()
    plt.xlabel('EPOCHS')
    plt.ylabel("")
    for k in mydict.keys():
        if len(mydict[k]) > 0 and k != "confusion":
            plt.plot(list(range(1, len(mydict[k]) + 1)), mydict[k], label=k)
    plt.title(model_name)
    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.savefig(mystring + '.png')

    # if "confusion" in mydict.keys():
    #     disp = ConfusionMatrixDisplay(confusion_matrix=mydict['confusion'][-1])
    #     disp.plot()
    #     plt.savefig('confusion_' + mystring + '.png')


def logthem(mytuple, name):
    if len(name) == 1 and len(mytuple) > 1:
        name = [name] * len(mytuple)
    elif len(name) == len(mytuple):
        pass
    else:
        print("We are not printing the keys or the name")
    counter = 0
    for x in mytuple:
        if isinstance(x, str):
            print("{}: {}".format(name[counter], x))
        elif isinstance(x, list):
            print("{}: {}".format(name[counter], x[0]))
        elif isinstance(x, set):
            print("{}: {}".format(name[counter], list(x[0])))
        elif isinstance(x, int):
            print("{}: {}".format(name[counter], str(x)))
        elif x is None:
            print("{}: None".format(name[counter]))
        counter += 1


def clean_label(label):
    label = label.replace("metric_dict_", "").replace("sentence-transformers_all-MiniLM-L12-v2", "_MiniLM")
    label = label.replace("roberta-base", "_roberta").replace("__", "_").replace("-", "").replace(".pkl", "")
    label = re.sub("_\d\d", "", label)
    return re.sub("_(.*)", r"\1", label)


def plot_comparison(list_pkls, ignore=["confusion"], dataset="", save_name=""):
    total_dict = {}
    for case in list_pkls:
        path = join(RESULTS, case + '.pkl') if 'pkl' not in case else join(RESULTS, case)
        with open(path, 'rb') as f:
            y = pickle.load(f)
            total_dict[case] = y
    list_keys = list(
        set(k for i in range(len(list_pkls)) for k in total_dict[list_pkls[i]] if total_dict[list_pkls[0]][k]))
    for metric in list_keys:
        plt.figure()
        plt.xlabel('EPOCHS')
        plt.ylabel(metric)
        for case in total_dict.keys():
            if not all(x in case for x in ignore) and isinstance(case, list):
                try:
                    yo = total_dict[case][metric]
                    label = clean_label(case).replace(dataset, "")
                    plt.plot(list(range(1, len(yo) + 1)), yo, label=label)
                except:
                    print("case", case)
                    print("total_dict[case]", total_dict[case])
        plt.title(metric.upper())
        plt.legend()
        plt.show()
        plt.tight_layout()
        plt.savefig(metric + "_" + dataset + save_name + '.png')


def deletefolder(folder):
    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clean_trained():
    from src.config import PHASE1_TRAIN, PHASE2_TRAIN, PHASE3_TRAIN, PHASE3_ONLY_TRAIN, MULTI_PRETRAIN
    list_of_folder = [PHASE1_TRAIN, PHASE2_TRAIN, PHASE3_TRAIN,
                      PHASE3_ONLY_TRAIN, MULTI_PRETRAIN]
    for folder in list_of_folder:
        deletefolder(folder)