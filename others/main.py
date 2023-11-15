from huggingface_hub import snapshot_download
import torch
import os
from os.path import basename
from itertools import product

# ############################################
import requests
# Monkey patch the requests functions
from functools import partial

# Monkey patch the requests functions
from src.multiple_trainings import multiple_training
from src.tools import clean_trained

requests.request = partial(requests.request, verify=False)
requests.get = partial(requests.get, verify=False)
requests.head = partial(requests.head, verify=False)
requests.post = partial(requests.post, verify=False)
requests.put = partial(requests.put, verify=False)
requests.patch = partial(requests.patch, verify=False)
requests.delete = partial(requests.delete, verify=False)
# Remove warning
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_mode = False

if __name__ == '__main__':
    from src.config import *

    torch.cuda.empty_cache()

    torch.manual_seed(42)
    import logging

    logging.basicConfig(filename=join("Data", "log.log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.info("3 steps approach in a loop.")
    logger = logging.getLogger(__name__)
    # Loop over the models
    list_extra_names = []
    # average = "macro"
    freezing = [True]  # False, True
    explode = [False]  # False, True
    length_ = [(True, True)]
    batch_ = [64]  # [2,6,8,12,16,32,64,128]
    P = [(12, 12)]
    LR_1_2 = [1e-04]  # 1e-04, 5e-05, 1e-5, 1e-6
    del_ratio = [0.4, 0.5, 0.6, 0.7]  # 0.3, 0.4, 0.5, 0.6, 0.7
    LR_3 = [2e-5, 1e-04, 1e-5]  # 1e-04, 1e-5, 1e-6
    for case in list(product(LR_1_2, P, batch_, length_, freezing, del_ratio, explode)):
        EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = case[1][0], 1, 40
        PHASE1_LR, PHASE2_LR, PHASE3_LR = case[0], case[0], 1e-05
        PHASE2_AUGMENTATION, PHASE3_eps = case[1][1], 2e-5
        PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = case[2], case[2], case[2], case[2]
        given_max_len, use_max_len = case[3][0], case[3][1]
        freezing = case[4]
        del_ratio = case[5]
        explode = case[6]
        for encoder in [
            (ROBERTA_BASE, "transformer", ["short", "medium"]),
            (ALL_MINI, "stransformer", ["short"])  # (BIG_BIRD_BASE, "transformer", ["long"])
        ]:
            if not os.path.exists(join(BASE_MODELS, encoder[0])):
                cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                              cache_dir="..",
                                              ignore_regex=["*.msgpack", "*.h5", "*.ot"])
                os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
            encoder_path = join(BASE_MODELS, encoder[0])
            # dataset = (path_phase1, path_phase2, val_path, test_path, ML_type,
            # size_of_the_samples_to_match_the_models, level)
            for dataset in [
                # (SNIPS_1, SNIPS_2, None, None, "classification", ["short", "medium"], 1, "macro", 48),
                # (SST2_1, SST2_2, None, SST2_test, "classification", ["short", "medium"], 1, "binary", 108),
                # (SNIPS2_1, SNIPS2_2, SNIPS2_val_2, SNIPS2_test, "classification",
                # ["short", "medium"], 1, "macro", 108),
                (SST5_1, SST5_2, SST5_val_2, SST5_test, "classification", ["short", "medium"], 1, "macro", 108),
                # SMALL BATCH
                # (AGNEWS_1, AGNEWS_2, None, AGNEWS_test, "classification", ["short", "medium"], 1, "macro", 320),
                # (IMDB_1, IMDB_2, None, IMDB_test, "classification", ["long", "medium"], 1, "binary", 500),
            ]:
                MAX_LEN = dataset[8] if given_max_len else 0
                long_mode = True if MAX_LEN >= 500 else False
                PHASE1_DECODER = encoder_path
                HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
                         PHASE1_LR, PHASE2_LR, PHASE3_LR,
                         PHASE2_AUGMENTATION, PHASE3_eps,
                         PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
                         MAX_LEN, PHASE1_DECODER, use_max_len, freezing,
                         del_ratio, explode, given_max_len]
                torch.cuda.empty_cache()
                multiple_training(encoder, dataset, encoder_path, HYPER,
                                  test_mode=test_mode, long_mode=long_mode,
                                  unbalance="normal")
                clean_trained()
