from os.path import join, exists
from os import makedirs
from pathlib import Path

# ---- BASE PATHS  ----
BASE_PATH = Path(__file__).parents[1].__str__()

BASE_DATA_PATH = join(BASE_PATH, "Data")
RESULTS = join(BASE_DATA_PATH, "RESULTS")
PHASE1 = join(BASE_DATA_PATH, "PHASE1")
PHASE2 = join(BASE_DATA_PATH, "PHASE2")
MODELS_PATH = join(BASE_PATH, "models")
APPSTOP = join(BASE_DATA_PATH, "STOPW/appstopwords.txt")

# #########################    DATA    ###########################
# ----- NOT FROM HUGGING FACE DATASET ----
SNIPS2 = join(BASE_DATA_PATH, "SNIPS2")
SNIPS2_TRAIN = join(SNIPS2, "train.csv")
SNIPS2_TEST = join(SNIPS2, "test.csv")
SNIPS2_DEV = join(SNIPS2, "dev.csv")

# ----- PHASE 1 ----
SST2_1 = join(PHASE1, "phase_1_sst2.csv")
SNIPS_1 = join(PHASE1, "phase_1_snips.csv")
SNIPS2_1 = join(PHASE1, "phase_1_snips2.csv")
IMDB_1 = join(PHASE1, "phase_1_imdb.csv")
YELP_1 = join(PHASE1, "phase_1_yelp.csv")
AGNEWS_1 = join(PHASE1, "phase_1_agnews.csv")
NEWS20_1 = join(PHASE1, "phase_1_20news.csv")
SST5_1 = join(PHASE1, "phase_1_sst5.csv")

# ----- PHASE 2 ----
SST2_2 = join(PHASE2, "phase_2_sst2.csv")
SNIPS_2 = join(PHASE2, "phase_2_snips.csv")
SNIPS2_2 = join(PHASE2, "phase_2_snips2.csv")
IMDB_2 = join(PHASE2, "phase_2_imdb.csv")
YELP_2 = join(PHASE2, "phase_2_yelp.csv")
AGNEWS_2 = join(PHASE2, "phase_2_agnews.csv")
NEWS20_2 = join(PHASE2, "phase_2_20news.csv")
SST5_2 = join(PHASE2, "phase_2_sst5.csv")

# ----- TEST ----
AGNEWS_test = join(PHASE2, "test_agnews.csv")
NEWS20_test = join(PHASE2, "test_NEWS20.csv")
YELP_test = join(PHASE2, "test_yelp.csv")
SST5_val_1 = join(PHASE2, "val_1_sst5.csv")
SST5_val_2 = join(PHASE2, "val_2_sst5.csv")
SST5_test = join(PHASE2, "test_sst5.csv")
SST2_val_1 = join(PHASE2, "val_1_sst2.csv")
SST2_val_2 = join(PHASE2, "val_2_sst2.csv")
SST2_test = join(PHASE2, "test_sst2.csv")
IMDB_test = join(PHASE2, "test_imdb.csv")
SNIPS2_val_1 = join(PHASE2, "val_1_snips2.csv")
SNIPS2_val_2 = join(PHASE2, "val_2_snips2.csv")
SNIPS_val_1 = join(PHASE2, "val_1_snips.csv")
SNIPS_test = join(PHASE2, "test_snips.csv")
SNIPS_val_2 = join(PHASE2, "val_2_snips.csv")
SNIPS2_test = join(PHASE2, "test_snips2.csv")

# #########################    MODELS    ###########################
# ----- MODELS ----
BASE_MODELS = join(MODELS_PATH, "base_models")
PHASE1_TRAIN = join(MODELS_PATH, "phase_1")
PHASE2_TRAIN = join(MODELS_PATH, "phase_2")
PHASE3_TRAIN = join(MODELS_PATH, "phase_3")
PHASE3_ONLY_TRAIN = join(MODELS_PATH, "phase_3_only")
MULTI_PRETRAIN = join(MODELS_PATH, "multi_pretrain")
PHASE3_MULTI_PRETRAIN = join(MODELS_PATH, "multi_pretrain_phase3")

# ----- BASE MODELS ----
BERT_BASE = "bert-base-uncased"
ROBERTA_BASE = "roberta-base"
ALL_MINI = 'sentence-transformers_all-MiniLM-L12-v2'
ALL_MPNETV2 = "sentence-transformers_all-mpnet-base-v2"
BIG_BIRD_LARGE = "google_bigbird-roberta-large"
BIG_BIRD_BASE = "bigbird-roberta-base"

# #########################    HYPERPARAMETERS    ###########################
# Hyperparameters PHASE 1
MAX_LEN = 0
PHASE1_BATCH = 8
EPOCHS_PHASE1 = 4
PHASE1_LR = 2e-5
PHASE1_ENCODER = ROBERTA_BASE
PHASE1_DECODER = ROBERTA_BASE
PHASE1_DATA = SST2_1

# Hyperparameters PHASE 2
PHASE2_BATCH = 8
EPOCHS_PHASE2 = 1
PHASE2_AUGMENTATION = 2
PHASE2_LR = 2e-5
PHASE2_MODEL = ROBERTA_BASE
PHASE2_DATA = SST2_2

# Hyperparameters PHASE 3
PHASE3_BATCH = 8
PHASE3_LR = 2e-5
PHASE3_eps = 1e-8
PHASE3_EPOCHS_MIN = 5
PHASE3_EPOCHS_MAX = 70

# Hyperparameters MULTI
MAX_LEN_MULTI = 0
MULTI_BATCH = 8
EPOCHS_MULTI = EPOCHS_PHASE2
MULTI_LR = PHASE2_LR
MULTI_TRAIN = join(MODELS_PATH, "phase_multi")


def create_tree():
    list_of_folder = [BASE_DATA_PATH, PHASE2, PHASE1, MODELS_PATH, BASE_MODELS, PHASE1_TRAIN,
                      PHASE2_TRAIN, PHASE3_TRAIN, PHASE3_ONLY_TRAIN, MULTI_PRETRAIN, RESULTS]
    for folder in list_of_folder:
        if not exists(folder):
            makedirs(folder)

