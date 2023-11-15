from huggingface_hub import snapshot_download
import torch
import os
from os.path import basename
# ############################################
import requests
# Monkey patch the requests functions
from functools import partial

# Monkey patch the requests functions
from src.ablations import multiple_training_CLFT

requests.request = partial(requests.request, verify=False)
requests.get = partial(requests.get, verify=False)
requests.head = partial(requests.head, verify=False)
requests.post = partial(requests.post, verify=False)
requests.put = partial(requests.put, verify=False)
requests.patch = partial(requests.patch, verify=False)
requests.delete = partial(requests.delete, verify=False)
# Remove warning
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    logging.info("No unbalance approach in a loop.")
    logger = logging.getLogger(__name__)
    # Loop over the models
    list_extra_names = []

    # ############################## ----- NORMAL UNBALANCE CORRECTION ----- ##############################

    # # ####################################   SNIPS2   #############################################
    dataset = (SNIPS2_1, SNIPS2_2, SNIPS2_val_2, SNIPS2_test, "classification", ["short", "medium"], 1, "macro", 108)
    encoder = (ALL_MINI, "stransformer", ["short"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 1e-5, 1e-5, 1e-5
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 4, 1, 70
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 32, 32, 32, 32
    PHASE2_AUGMENTATION, PHASE3_eps = 12, 2e-5
    use_max_len, given_max_len, freezing, del_ratio = True, True, False, 0.3
    MAX_LEN = dataset[8] if given_max_len else 0
    PHASE1_DECODER = encoder_path
    doexplode = False
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")

    # # ###################################   AGNEWS   ########################################
    dataset = (AGNEWS_1, AGNEWS_2, None, AGNEWS_test, "classification", ["short", "medium"], 1, "macro", 320)
    encoder = (ROBERTA_BASE, "transformer", ["short", "medium"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 4, 1, 70
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 2e-04, 2e-05, 2e-05
    PHASE2_AUGMENTATION, PHASE3_eps = 4, 2e-5
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 16, 16, 16, 16
    use_max_len, given_max_len, freezing, del_ratio = False, False, True, 0.6
    MAX_LEN = dataset[8] if given_max_len else 0
    PHASE1_DECODER = encoder_path
    doexplode = False
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")

    #  ######################### IMDB ############################
    dataset = (IMDB_1, IMDB_2, None, IMDB_test,
               "classification", ["long", "medium"], 1, "binary", None)
    encoder = (ROBERTA_BASE, "transformer", ["short", "medium"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 1e-5, 1e-5, 1e-5
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 2, 1, 50
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 6, 6, 6, 6
    PHASE2_AUGMENTATION, PHASE3_eps = 4, 2e-5
    use_max_len, given_max_len, freezing, del_ratio = True, True, False, 0.6
    MAX_LEN = 512
    PHASE1_DECODER = encoder_path
    doexplode = True
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")

    # # ######################################   SNIPS   ##########################################
    dataset = (SNIPS_1, SNIPS_2, None, None, "classification", ["short", "medium"], 1, "macro", 48)
    encoder = (ALL_MINI, "stransformer", ["short"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 5e-5, 5e-5, 5e-5
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 4, 1, 70
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 32, 32, 32, 32
    PHASE2_AUGMENTATION, PHASE3_eps = 12, 2e-5
    use_max_len, given_max_len, freezing, del_ratio = True, False, False, 0.6
    MAX_LEN = dataset[8] if given_max_len else 0
    PHASE1_DECODER = encoder_path
    doexplode = False
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")

    # # ######################################   SST2   #######################################
    dataset = (SST2_1, SST2_2, None, SST2_test, "classification", ["short", "medium"], 1, "binary", 108)
    encoder = (ROBERTA_BASE, "transformer", ["short", "medium"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 1e-5, 1e-5, 1e-5
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 12, 1, 70
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 64, 64, 64, 64
    PHASE2_AUGMENTATION, PHASE3_eps = 12, 2e-5
    use_max_len, given_max_len, freezing, del_ratio = False, False, False, 0.6
    MAX_LEN = dataset[8] if given_max_len else 0
    PHASE1_DECODER = encoder_path
    doexplode = False
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")
    #
    # # ###################################   SST5   ######################################
    dataset = (SST5_1, SST5_2, SST5_val_2, SST5_test, "classification", ["short", "medium"], 1, "macro", 108)
    encoder = (ROBERTA_BASE, "transformer", ["short", "medium"])
    if not os.path.exists(join(BASE_MODELS, encoder[0])):
        cache_dir = snapshot_download(repo_id=basename(encoder[0]).replace("_", "/"),
                                      cache_dir="..",
                                      ignore_regex=["*.msgpack", "*.h5", "*.ot"])
        os.rename(cache_dir, join(BASE_MODELS, encoder[0]))
    encoder_path = join(BASE_MODELS, encoder[0])
    PHASE1_LR, PHASE2_LR, PHASE3_LR = 1e-5, 1e-5, 1e-5
    EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX = 12, 1, 70
    PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH = 64, 64, 64, 64
    PHASE2_AUGMENTATION, PHASE3_eps = 12, 2e-5
    use_max_len, given_max_len, freezing, del_ratio = False, False, False, 0.6
    MAX_LEN = dataset[8] if given_max_len else 0
    PHASE1_DECODER = encoder_path
    doexplode = False
    HYPER = [EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
             PHASE1_LR, PHASE2_LR, PHASE3_LR,
             PHASE2_AUGMENTATION, PHASE3_eps,
             PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
             MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
             doexplode, given_max_len]
    torch.cuda.empty_cache()
    multiple_training_CLFT(encoder, dataset, encoder_path, HYPER,
                           list_extra_names=list_extra_names, unbalance="normal")
