from src.config import SNIPS_1, SNIPS_2, SST2_1, SST2_2, SNIPS_DATA
from src.config import IMDB_1, IMDB_2, SST2_DATA, IMDB_DATA

import argparse
import os
import pandas as pd
import json
import pickle
from sklearn.utils import shuffle

# import torch
# torch.cuda.empty_cache()

parser = argparse.ArgumentParser(
    "Select mode either test or train to choose if we want the datasets or a small portion")
parser.add_argument("mode", default="train",
                    help="decide either 'test' or 'train'", type=str)
args = parser.parse_args()

# #####################################################################
print("Preprocessing SST2")

# Loading data
df1 = pd.read_csv(
    os.path.join(SST2_DATA, 'SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt'), sep='|')
df1.columns = ['query', 'phrase ids']
df2 = pd.read_csv(
    os.path.join(SST2_DATA, 'SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt'),
    sep='|')
df3 = df1.merge(df2, how='left', on='phrase ids')
df3 = df3.rename(columns={"sentiment values": "cat1"})

# Cleaning
df1["len"] = df1["query"].apply(lambda x: len(x.split()))
max_no_words = df1["len"].max()

# Saving
ser1, SST2 = df3["query"], df3[["query", "cat1"]]
ser1, SST2 = shuffle(ser1), shuffle(SST2)

if args.mode == "test":
    ser1, SST2 = ser1[:100], SST2[:100]

ser1.to_csv(SST2_1, index=False)
SST2.to_csv(SST2_2, index=False)

# #####################################################################
print("Preprocessing SNIPS")

# Load data
# https://github.com/sonos/nlu-benchmark/blob/master/2016-12-built-in-intents/benchmark_data.json
with open(os.path.join(SNIPS_DATA, "benchmark_data.json"), encoding="utf8") as f:
    snips_dict = json.load(f)

snips_df = pd.DataFrame(
    [[query["text"], intent["benchmark"]["Snips"]["original_intent_name"]] for domain in snips_dict["domains"] for
     intent in domain["intents"] for query in intent["queries"]], columns=["text", "label"])
label2id = {label: i for i, label in enumerate(snips_df["label"].unique())}
snips_df["category"] = snips_df["label"].apply(lambda x: label2id[x])

snips_df["len"] = snips_df["text"].apply(lambda x: len(x.split()))
print("mean number of words: {}".format(int(snips_df["len"].mean())))
print("max number of words: {}".format(snips_df["len"].max()))

# Saving
snips_2 = snips_df[["text", "category"]]
snips_2 = snips_2.rename(columns={"text": "query", "category": "cat1"})
snips_1 = snips_2["query"]
snips_1, snips_2 = shuffle(snips_1), shuffle(snips_2)

if args.mode == "test":
    snips_2, snips_1 = snips_2[:100], snips_1[:100]

snips_1.to_csv(SNIPS_1, index=False)
snips_2.to_csv(SNIPS_2, index=False)

with open(os.path.join(SNIPS_DATA, "label2id.pkl"), "wb+") as file:
    pickle.dump(label2id, file)
    file.close()

# https://github.com/sonos/nlu-benchmark/blob/master/2016-12-built-in-intents/benchmark_data.json
with open(os.path.join(SNIPS_DATA, "benchmark_data.json"), encoding="utf8") as f:
    snips_dict = json.load(f)

snips_df = pd.DataFrame(
    [[query["text"], intent["benchmark"]["Snips"]["original_intent_name"]] for domain in snips_dict["domains"] for
     intent in domain["intents"] for query in intent["queries"]], columns=["text", "label"])
label2id = {label: i for i, label in enumerate(snips_df["label"].unique())}
snips_df["category"] = snips_df["label"].apply(lambda x: label2id[x])

# #####################################################################
print("Preprocessing IMDB")

# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz (Unzip)
imdb_list = [[open(os.path.join(IMDB_DATA, "aclImdb", split, label, filename),
                   encoding="utf8").read(), split, label, name] for split in ["train", "test"] for label, name in
             [("pos", 1), ("neg", 0), ("unsup", -1)] if os.path.isdir(os.path.join(IMDB_DATA, "aclImdb", split, label))
             for filename in os.listdir(os.path.join(IMDB_DATA, "aclImdb", split, label))]
imdb_df = pd.DataFrame(imdb_list, columns=["text", "split", "label", "category"])
imdb_df = imdb_df.rename(columns={"text": "query", "category": "cat1"})
imdb_df_train = imdb_df[imdb_df["split"] == "train"]
imdb_df_test = imdb_df[imdb_df["split"] == "test"]

df = imdb_df_train[imdb_df_train["query"].str.contains("<")].reset_index()

imdb_df_train["len"] = imdb_df_train["query"].apply(lambda x: len(x.split()))
print("mean number of words: {}".format(int(imdb_df_train["len"].mean())))
print("max number of words: {}".format(imdb_df_train["len"].max()))

imdb_2, imdb_1 = imdb_df_train[["query", "cat1"]], imdb_df_train["query"]
imdb_2, imdb_1 = shuffle(imdb_2), shuffle(imdb_1)

if args.mode == "test":
    imdb_2, imdb_1 = imdb_2[:100], imdb_1[:100]
imdb_1.to_csv(IMDB_1, index=False)
imdb_2.to_csv(IMDB_2, index=False)
