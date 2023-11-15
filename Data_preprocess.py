from src.config import SNIPS_1, SNIPS_2, SST2_1, SST2_2, YELP_1, YELP_2, YELP_test
from src.config import SST5_1, SST5_2, SST5_val_1, SST5_val_2, SST5_test
from src.config import IMDB_1, IMDB_2, IMDB_test
from src.config import AGNEWS_1, AGNEWS_2, AGNEWS_test
from src.config import NEWS20_1, NEWS20_2, NEWS20_test
from src.config import SST2_val_1, SST2_val_2, SST2_test
from src.config import create_tree

import torch
from datasets import load_dataset
from sklearn import preprocessing as prep
import numpy as np
import argparse
import re
import pandas as pd
import requests
from sklearn.utils import shuffle
from functools import partial

create_tree()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(
    "Select mode either test or train to choose if we want the datasets or a small portion")
parser.add_argument("mode", default="train",
                    help="decide either 'test' or 'train'", type=str)

args = parser.parse_args()
test_mode = True if args.mode == "test" else False

# ############################################
import requests
# Monkey patch the requests functions
from functools import partial

# Monkey patch the requests functions
requests.request = partial(requests.request, verify=False)
requests.get = partial(requests.get, verify=False)
requests.head = partial(requests.head, verify=False)
requests.post = partial(requests.post, verify=False)
requests.put = partial(requests.put, verify=False)
requests.patch = partial(requests.patch, verify=False)
requests.delete = partial(requests.delete, verify=False)
# Remove warning
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# ############################################


def huggingface_to_ours(path="rungalileo/20_Newsgroups_Fixed", test_mode=False, hierarchycal=1, join_train_val=False):
    raw_datasets = load_dataset(path)

    CLEANR = re.compile('<.{0,6}>')

    def cleanhtml(raw_html):
        return re.sub(CLEANR, '', raw_html)

    def cleantext(raw_text):
        return raw_text.replace("\t")

    # Dataframe preparation
    datasets = []
    print("KEYS: ", list(raw_datasets.keys()))
    for k in raw_datasets.keys():
        ds = raw_datasets[k].to_pandas()
        if join_train_val:
            ds["split"] = "test" if k == "test" else "train"
        else:
            ds["split"] = k
        datasets.append(ds)
    df = pd.concat(datasets).reset_index(drop=True)
    print("Label", list(df['label'].unique()))
    if hierarchycal > 1:
        df[['cat1', 'cat2']] = df['label'].str.split('.', 1, expand=True)
        del df['label']
    df = df.rename(columns={"label": "cat1", "text": "query", "sentence": "query"})

    # Encoding
    categories = [x for x in df.columns if 'cat' in x]
    for cat in categories:
        tag_encoded = prep.LabelEncoder()
        if not isinstance(df[cat][0], str):
            df_sub = df[df[cat]!=-1]
            df_sub[cat] = tag_encoded.fit_transform(df_sub[cat])
            df[df[cat]!=-1][cat] = df_sub[cat]
        else:
            df[cat] = tag_encoded.fit_transform(df[cat])
    df["query"] = df["query"].apply(lambda x: x.replace("\n", "") if isinstance(x, str) else np.nan)
    df.dropna(inplace=True)
    df['query'] = df['query'].map(cleanhtml)

    # Recovering
    outputs = dict()
    for k in raw_datasets.keys():
        df_sub = df[df["split"] == k]
        df_sub = shuffle(df_sub)
        if test_mode:
            df_sub = df_sub[:200]
        df_sub2 = df_sub[["query"] + categories]
        df_sub1 = df_sub["query"]
        outputs[k + "1"] = df_sub1
        outputs[k + "2"] = df_sub2
    return outputs


# ############################################ SST-2
sst2 = huggingface_to_ours(path="sst2", test_mode=test_mode, join_train_val=False)

sst2['train1'] = pd.concat([sst2["test1"], sst2["train1"]]).reset_index(drop=True)
sst2['train1'].to_csv(SST2_1, index=False)
sst2['train2'].to_csv(SST2_2, index=False)
sst2['validation2'].to_csv(SST2_test, index=False)

# ############################################ snips
snips = huggingface_to_ours(path="snips_built_in_intents", test_mode=test_mode)
snips['train1'].to_csv(SNIPS_1, index=False)
snips['train2'].to_csv(SNIPS_2, index=False)

# ############################################ 20 news
news220 = huggingface_to_ours(path="rungalileo/20_Newsgroups_Fixed", test_mode=test_mode, hierarchycal=2)
news220['train1'] = news220['train1'].apply(lambda x: " " if len(x) == 0 else x)
news220['train2']['query'] = news220['train2']['query'].apply(lambda x: " " if len(x) == 0 else x)
news220['test2']['query'] = news220['test2']['query'].apply(lambda x: " " if len(x) == 0 else x)
news220['train1'].to_csv(NEWS20_1, index=False)
news220['train2'].to_csv(NEWS20_2, index=False)
news220['test2'].to_csv(NEWS20_test, index=False)

# ############################################ Yelp
# Yelp ratings
yelp = huggingface_to_ours(path="yelp_review_full", test_mode=test_mode)
yelp['train1'].to_csv(YELP_1, index=False)
yelp['train2'].to_csv(YELP_2, index=False)
yelp['test2'].to_csv(YELP_test, index=False)

# ############################################ SST5
# Rating
sst5 = huggingface_to_ours(path="SetFit/sst5", test_mode=test_mode)
sst5['train1'].to_csv(SST5_1, index=False)
sst5['train2'].to_csv(SST5_2, index=False)
sst5['validation1'].to_csv(SST5_val_1, index=False)
sst5['validation2'].to_csv(SST5_val_2, index=False)
sst5['test2'].to_csv(SST5_test, index=False)

sst5['train2']['len'] = sst5['train2']["query"].apply(lambda x: len(x.split(" ")))
print("mean number of words: {}".format(int(sst5['train2']["len"].mean())))
print("max number of words: {}".format(sst5['train2']["len"].max()))

# ############################################ AG news
ag_news = huggingface_to_ours(path="ag_news", test_mode=test_mode)
ag_news['train1'].to_csv(AGNEWS_1, index=False)
ag_news['train2'].to_csv(AGNEWS_2, index=False)
ag_news['test2'].to_csv(AGNEWS_test, index=False)
# this version of the data has the title and the description connected with a space

# ############################################ IMDB
imdb = huggingface_to_ours(path="imdb", test_mode=test_mode)
imdb_1 = pd.concat([imdb["train1"], imdb["unsupervised1"]]).reset_index(drop=True)

imdb_1.to_csv(IMDB_1, index=False)
imdb["train2"].to_csv(IMDB_2, index=False)
imdb['test2'].to_csv(IMDB_test, index=False)

imdb_1["len"] = imdb_1.apply(lambda x: len(x.split()))
print("mean number of words: {}".format(int(imdb_1["len"].mean())))
print("max number of words: {}".format(imdb_1["len"].max()))
# ############################################
