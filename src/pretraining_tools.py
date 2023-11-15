import math

import pandas as pd
from torch.utils.data import DataLoader
from sklearn import preprocessing as prep
from sklearn.utils import shuffle
from sentence_transformers import evaluation, InputExample

from src.config import PHASE2_BATCH
from src.data_augmentation import score_balancer

import logging
logger = logging.getLogger(__name__)
# ##################  SCORING FUNCTION  ##################


def scoring_hierarchical(**kwargs):
    """ This function creates the labels for classification in the hierarchical,
        whenever we have different categories and subcategories, as well as the simple.
        considering an only label.
        There are three types of keys per level. For level N:
            cat1_N (ref to query 1 level N)
            cat2_N (ref to query 2 level N)
            wN (ref to the weight of the level N)
    """
    levels = int(len(kwargs) / 3)
    score = 0
    for i in range(levels):
        no = str(i+1)
        score += kwargs["w" + no] if kwargs["cat1_" + no] == kwargs["cat2_" + no] else 0
    return float(score)


def scoring_regression(**kwargs):
    """ This function creates the labels for the simple or the hierarchical case of regression.
        There are two types of keys per level. For level N:
            cat1_N (ref to query 1 level N)
            cat2_N (ref to query 2 level N)
    The scores must be normalized in advance
    """
    levels = int(len(kwargs) / 2)
    score = 0
    weight = 1 / levels
    for i in range(levels):
        no = str(i + 1)
        score += (1 - abs(kwargs["cat1_" + no] - kwargs["cat2_" + no])) / weight
    return float(score)


def scoring(df, model_type, *weights):
    """ This function creates a column "score" based on a score weighted by the weights"""
    # logger.debug("Input dadataframe for scoring {}".format(df.head(1)))
    model_type = model_type.lower()
    # Classification case
    if "class" in model_type:

        def create_input(row, *args):
            output = dict()
            for x in range(len(args)):
                no = str(x+1)
                output["cat1_" + no] = row["cat1_" + no]
                output["cat2_" + no] = row["cat2_" + no]
                output["w" + no] = args[x]
            return output

        df["score"] = df.apply(lambda x: scoring_hierarchical(**create_input(x, *weights)), axis=1)

    # Regression case
    elif "reg" in model_type:
        list_cat = [k for k in df.columns if "cat" in k]
        df["score"] = df.apply(lambda x: scoring_regression(**{k: x[k] for k in list_cat}), axis=1)
    df = df[["score", "sentence1", "sentence2"]]
    # logger.debug("Head of the resulting DataFrame", df.head(1))
    logger.debug("Max {} and min {} values of the scores".format(df["score"].max(), df["score"].min()))
    return df


# ##################  ENCODING  ##################

def encoding_classification(df, hierarchy_level=1):
    """ Encoding function for classification"""
    for i in range(hierarchy_level):
        # Encoder
        tag_encoded = prep.LabelEncoder()
        # using fit_transform we standardize the distributions of tag
        df["cat" + str(i + 1)] = tag_encoded.fit_transform(df["cat" + str(i + 1)])
    print("cat" + str(hierarchy_level) + " counts", df["cat" + str(hierarchy_level)].value_counts())
    return df


# ##################  SPLIT  ##################

def train_test_spliter(total, split, evaluation_data=False):
    """ This function just split whenever we do not have an eval dataset. """
    if evaluation_data:
        return total, evaluation_data
    else:
        train = total.sample(frac=split, random_state=42)
        test = total.drop(train.index)
        return train, test


# #######################   OTHERS    #######################

def sbert_dataloader(df, BATCH):
    """ This function creates the dataloader for sbert training with the sbert library format"""
    training_samples = list(map(lambda x: InputExample(texts=[x[0], x[1]], label=x[2]),
                                list(zip(df['sentence1'].tolist(),
                                         df['sentence2'].tolist(),
                                         df['score'].tolist()))))
    train_dataloader = DataLoader(training_samples, shuffle=True, batch_size=BATCH)
    return train_dataloader


def create_false(phase_2_ids, rate):
    """ This function starts with a Dataframe with ids, sent_1, sent_2 and 1 as similarity
        and creates inside the set of indexes false samples giving as output a dataframe
        with rate times the amount of positives.
        This was done for semi-supervised contrastive """
    # List of ids to take into consideration
    list_ids = list(set(phase_2_ids["sent_1"].to_list()))

    # Rates computation
    positive = phase_2_ids.shape[0] / 2
    negative = positive * rate
    subrate = round(negative / len(list_ids))

    # Loop to create the negative samples
    cols = list(phase_2_ids.columns)
    falses = pd.DataFrame(columns=cols)
    sent_1, sent_2 = [], []
    for myid in list_ids:
        subdf = phase_2_ids[(phase_2_ids["sent_1"] != myid) & (phase_2_ids["sent_2"] != myid)]
        for i in range(subrate):
            myrow = subdf.sample()
            sent_1.append(myid)
            sent_2.append(myrow.iloc[0, 1])
    falses[cols[0]] = sent_1
    falses[cols[1]] = sent_2
    falses["score"] = [0] * len(sent_2)
    return phase_2_ids.append(falses, ignore_index=True)


def evaluator_(df, evaluation_metric, model_type="classification", *weights):
    """ This function creates an evaluator for the library sbert. w1 and w2 (btn 0 and 1) are the weights
    to create the score. evaluation_metric ("sim" or "binary" is the type of evaluator)"""
    evaluation_metric = evaluation_metric.lower()
    evaluation_metric = "sim" if "sim" in evaluation_metric else "bin" if "bin" in evaluation_metric else "mse"

    if "bin" == evaluation_metric and "class" in model_type:
        weights = [0] * (len(weights) - 1) + [weights[-1]]

    if any([x in evaluation_metric for x in ["sim", "bin"]]):
        # ----- SCORING
        df = scoring(df, model_type, *weights)

        # ----- Balance the dataset or round to 0 and 1 if bin
        if evaluation_metric == "bin":
            if "class" in model_type:
                counts = df["score"].value_counts()
                threshold = counts.min()
                df = score_balancer(df, counts, threshold, mode="evaluator")
            else:
                df['score'] = df['score'].apply(lambda x: round(x))

        # ---- Evaluator
        if evaluation_metric == "bin":
            evaluator = evaluation.BinaryClassificationEvaluator(df['sentence1'].tolist(),
                                                                 df['sentence2'].tolist(),
                                                                 df['score'].tolist())
        else:
            evaluator = evaluation.EmbeddingSimilarityEvaluator(df['sentence1'].tolist(),
                                                                df['sentence2'].tolist(),
                                                                df['score'].tolist())
        return evaluator

    elif "mse" in evaluation_metric.lower():
        return None


def combinations(df, dataset_size, multiply_by, level):
    """ This function creates pairs of data, two sentences and their values for levels 1, 2 and
    ,if it exits, 3. For a df with n rows we can create n^2 pairs of sentences, this function creates
    multiply_by * n pairs of sentences. It also filters by percentage of those sentences"""
    # Column names
    query_col = [col for col in df.columns.tolist() if all([x not in col for x in ["cat", "score"]])][0]
    labels_num = len(df["cat" + str(level)].unique())
    multiply_by = multiply_by * int(math.log(labels_num, 2)) if labels_num < 100 else multiply_by
    final_cols = [query_col] + ["cat" + str(i + 1) for i in range(level)]
    df = df[final_cols]

    # Creating dataset by combinations
    evaluation1 = pd.concat([shuffle(df) for _ in range(multiply_by)]).reset_index(drop=True)
    evaluation2 = pd.concat([shuffle(df) for _ in range(multiply_by)]).reset_index(drop=True)

    evaluation1.columns = ["sentence1"] + ["cat1_" + str(i + 1) for i in range(level)]
    evaluation2.columns = ["sentence2"] + ["cat2_" + str(i + 1) for i in range(level)]

    evaluation3 = pd.concat([evaluation2, evaluation1], axis=1)
    del evaluation1, evaluation2

    # SIZE
    if dataset_size == "small":
        evaluation3 = evaluation3.sample(frac=0.1).reset_index(drop=True)
    elif dataset_size == "medium":
        evaluation3 = evaluation3.sample(frac=0.25).reset_index(drop=True)
    elif dataset_size == "big":
        evaluation3 = evaluation3.sample(frac=0.5).reset_index(drop=True)
    elif dataset_size == "extra_big":
        evaluation3 = evaluation3.sample(frac=1).reset_index(drop=True)
    return evaluation3
