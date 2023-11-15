from src.DAE import load_tsdae_data, tsdae_evaluator, tsdae_train_dataset, max_len_wrapper, dae_trainer
from src.head import *
from src.config import *
from src.tools import check_dir, create_val_test_train, read_csv, chunk_it, trainer, optimization_setup, \
    save_and_plot, check_device, logthem, plot_comparison, text_split
from src.pretraining_tools import encoding_classification
from src.CL import training_sbert, sbert_data_preproces
from sentence_transformers import evaluation, losses
from transformers import AdamW
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig

import os
from os.path import basename, isfile, join
from os import mkdir, listdir
# nltk.download('punkt')

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


def Just_FT(encoder, dataset, encoder_path, HYPER, test_mode=False, long_mode=False, list_extra_names=[]):
    """ One of the iterations in the loop for one of combination of hyperparameters.
    Input:
        - encoder (tuple): (models path: str, model type: str, lengths: list[str])
        - dataset (tuple):
                (path to dataset 1, path to dataset 2, path to validation dataset, path to test dataset,
                task type: "classification",lengths: list[str], level: if > 1 then hierarchical, metric,
                default MAX_LEN)
        - encoder_path (tuple):
        - HYPER (list): list of hyperparameters just unpackage below
    """
    (EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
     PHASE1_LR, PHASE2_LR, PHASE3_LR,
     PHASE2_AUGMENTATION, PHASE3_eps,
     PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
     MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
     doexplode, given_max_len) = HYPER
    if not set(encoder[2]).intersection(set(dataset[5])):
        return None
    else:
        print(f'Encoder length {encoder[2]}. Dataset length {dataset[5]}')
        length = set(encoder[2]).intersection(set(dataset[5]))

    # ####################################################################################
    # #
    # # ####################################  PHASE 0  ####################################
    # #
    # ####################################################################################

    print("\n##############  PHASE 0  ##############")
    metric_dict_main, metric_dict_jl, metric_dict_ft = None, None, None

    logthem(encoder, ["model name", "type of model", "size of model"])
    logthem(dataset, ["path_phase1", "path_phase2", "val_path", "test_path", "ML_type", "size",
                      'level', "average", "max_len"])

    # Data paths: first to a series with the strings and second to a dataframe
    # with the categories or values of the regression
    data_path_1 = dataset[0]
    data_path_2 = dataset[1]
    val_data_str = dataset[2]
    test_data_str = dataset[3]
    average = dataset[7]

    # Do we use validation to choose the best model? If yes, which type of evaluation from the library sbert
    evaluate = True
    evaluation_metric = "bin" if average == "binary" else "sim"

    # Type of base model's architecture
    base_architecture_type = encoder[1]

    # level of the hierarchy
    level = dataset[6]

    # If the model is for regression or for classification
    model_type = dataset[4]

    base_model_config = AutoConfig.from_pretrained(encoder_path)

    # # explode data
    sbert_data = read_csv(data_path_2)
    df_val = read_csv(val_data_str).dropna() if val_data_str else None
    df_test = read_csv(test_data_str).dropna() if test_data_str else None
    sbert_data = sbert_data.dropna()
    sbert_data = check_dtype(sbert_data)
    if val_data_str:
        df_val = check_dtype(df_val)
    if test_data_str:
        df_test = check_dtype(df_test)
    if long_mode:
        # Adapt data to the model
        # max_len_original = base_model_config.max_position_embeddings - 10
        print(f"-------------- max_len_original {MAX_LEN} ----------")
        tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        if doexplode:
            df1, df2 = chunk_it(sbert_data, tokenizer, MAX_LEN)
            _, df_test = chunk_it(df_test, tokenizer, MAX_LEN)
        else:
            sbert_data["query"] = sbert_data["query"].apply(lambda x: text_split(x, tokenizer, MAX_LEN)[0])
            df1, df2 = sbert_data["query"].copy(), sbert_data.copy()
        if df_val is not None: df_val["query"] = df_val["query"].apply(lambda x: text_split(x, tokenizer, MAX_LEN)[0])
        if df_test is not None: df_test["query"] = df_test["query"].apply(
            lambda x: text_split(x, tokenizer, MAX_LEN)[0])
    else:
        df1 = read_csv(data_path_1)
        df2 = sbert_data.copy()
    data_path_phase1 = join("Data", "temp_df1.csv")
    data_path_phase2 = join("Data", "temp_df2.csv")
    if df_val is not None: val_data_str = join("Data", "temp_val.csv")
    if df_val is not None: test_data_str = join("Data", "temp_test.csv")
    if test_mode:
        df1 = df1[:200]
        df2 = df2[:200]
    if test_data_str is not None: df_test.to_csv(test_data_str, index=False)
    if val_data_str is not None: df_val.to_csv(val_data_str, index=False)
    df1.to_csv(data_path_phase1, index=False)
    df2.to_csv(data_path_phase2, index=False)

    R = 1 if df1.shape[0] > 500 else 1 if test_mode else 0.2
    # R = 1
    BATCH1 = int(PHASE1_BATCH * R)
    BATCH2 = int(PHASE2_BATCH * R)
    BATCH3 = int(PHASE3_BATCH * R)

    # Print the hyperparameters
    list1 = [MAX_LEN, BATCH1, EPOCHS_PHASE1, PHASE1_LR, PHASE1_ENCODER, PHASE1_DECODER,
             BATCH2, EPOCHS_PHASE2, PHASE2_AUGMENTATION, PHASE2_LR, PHASE2_MODEL,
             BATCH3, PHASE3_LR, PHASE3_eps, PHASE3_EPOCHS_MAX, freezing, del_ratio, given_max_len]
    list2 = ["MAX_LEN", "PHASE1_BATCH", "EPOCHS_PHASE1", "PHASE1_LR", "PHASE1_ENCODER", "PHASE1_DECODER",
             "PHASE2_BATCH", "EPOCHS_PHASE2", "PHASE2_AUGMENTATION", "PHASE2_LR",
             "PHASE2_MODEL", "PHASE3_BATCH", "PHASE3_LR", "PHASE3_eps",
             "PHASE3_EPOCHS_MAX", "freezing", "del_ratio", "given_max_len"]
    parameters = dict(zip(list2, list1))
    for k, v in parameters.items():
        print(k, str(v))

    # Shared name for all the phases for this model
    eval_string = "eval" if evaluate else ""
    extra_name = "_LR" + str(PHASE1_LR) + str(PHASE2_LR) + str(PHASE3_LR) \
                 + "_EP" + str(EPOCHS_PHASE1) + str(EPOCHS_PHASE2) + str(PHASE3_EPOCHS_MAX) \
                 + "_BA" + str(PHASE1_BATCH) + str(PHASE2_BATCH) + str(PHASE3_BATCH) \
                 + "_eps" + str(PHASE3_eps) + "AUG" + str(PHASE2_AUGMENTATION) \
                 + "U" + str(1 if use_max_len else 0) + "G" + str(1 if given_max_len else 0) \
                 + "F" + str(1 if freezing else 0) + "D" + str(del_ratio) + "E" + str(1 if doexplode else 0)
    list_extra_names.append(extra_name)
    model_name = '{}_{}_{}_{}'.format(dataset[0].split(".")[0].split("_")[-1].upper(),
                                      eval_string,
                                      basename(encoder_path).replace("sentence-transformers", ""),
                                      extra_name)
    print(f"\n Number of examples ################ {df2.shape}\n")
    device = check_device()
    truncation = 512 if length == "medium" else 128 if length == "short" else 1024
    # ###################################################################################
    #
    # #################################  PURE Fine-tuning  #################################
    #
    # ###################################################################################

    print("\n##############  PURE PHASE 3  ##############")
    torch.cuda.empty_cache()

    # Configuration
    device = check_device()
    base_config_json = {
        "freezing": freezing,
        "device": device,
        "base_model_path": encoder_path,
        "embedding_size": base_model_config.hidden_size,
        "transformer": True if encoder[1] == "transformer" else False,
        "model_type": model_type,
        "truncation": truncation,
        "from_embeddings": False
    }
    base_config = BaseConfig(**base_config_json)

    # Data preprocess
    sbert_data = pd.read_csv(data_path_phase2)
    sbert_data = sbert_data.dropna(how="any")
    sbert_data = encoding_classification(sbert_data, hierarchy_level=level)

    # Create model for regression
    model = None
    if "reg" in model_type:
        model = Classifier_head(base_config, num_class=1, architecture="complex")
        model.to(device)

    # Create model for classification
    if "class" in model_type:
        model = Classifier_head(base_config, num_class=len(sbert_data["cat1"].unique()),
                                architecture="complex")
        model.to(device)

    train_dataloader, test_dataloader = create_val_test_train(test_data_str, sbert_data, model_type,
                                                              BATCH3)

    # Hyperparameters 2
    scheduler, optimizer = optimization_setup(train_dataloader, model, PHASE3_LR, PHASE3_eps, PHASE3_EPOCHS_MAX)
    train_loss, metric_dict_ft = trainer(PHASE3_EPOCHS_MIN, PHASE3_EPOCHS_MAX, train_dataloader,
                                         test_dataloader, model,
                                         optimizer, device, scheduler, model_type, average)

    final_3_only = join(PHASE3_ONLY_TRAIN, "PHASE3_" + model_name)
    try:
        mkdir(final_3_only)
    except OSError as error:
        print(error)
    saving_path_bin = join(final_3_only, "model.bin")
    check_dir(saving_path_bin)
    torch.save(model.state_dict(), saving_path_bin)

    save_name = '{}_{}_{}'.format(dataset[0].split(".")[0].split("_")[-1].upper(),
                                  basename(encoder_path),
                                  extra_name)
    save_and_plot(metric_dict_ft, "ft", save_name)

    for dataset in ["SST2"]:
        for extra in list_extra_names:
            onlyfiles = [f for f in listdir(RESULTS) if isfile(join(RESULTS, f))]
            list_pkls = [x for x in onlyfiles if dataset in x]
            list_pkls = [x for x in list_pkls if extra in x]
            plot_comparison(list_pkls, ignore=["confusion"], dataset=dataset, save_name=save_name)


def joint(model_type, data_path_phase1, data_path_phase2, encoder, level, encoder_path, use_max_len, BATCH1,
          del_ratio, length, BATCH2, val_data_str, eval_string, dataset, extra_name, base_config_json,
          test_data_str, base_model_config, average, BATCH3):
    print("\n##############  MULTITASK PHASE  ##############")
    torch.cuda.empty_cache()
    device = check_device()

    data_tsdae_path = data_path_phase1
    val_tsdae_path = data_path_phase2
    evaluate = True
    evaluation_metric = "bin"
    base_type = encoder[1]

    # ### Dataloader tsdae
    weights = [1 / level for _ in range(level)]

    # # Data sets: train just a series with text and val_data with labels
    # We use the epochs from the first phase to increase the data and the epochs for multi for the real training
    data_dae, val_data = load_tsdae_data(train_path=data_tsdae_path, val_path=val_tsdae_path)
    data_dae = pd.concat([data_dae] * EPOCHS_PHASE1, axis=0, ignore_index=True)

    # Model
    list_queries = data_dae["name_query"].to_list()
    model, max_len = max_len_wrapper(list_queries[::2].copy(), use_max_len,
                                     MAX_LEN, encoder_path, base_type, device)

    # Evaluator
    evaluator_tsdae = tsdae_evaluator(
        val_data,
        evaluation_metric,
        model_type,
        level,
        *weights
    )
    # Dataloader
    train_dataloader_tsdae = tsdae_train_dataset(list_queries, BATCH1, del_ratio)

    # ### Losses
    # It has a cross entropy loss in the back
    PHASE1_DECODER = encoder_path
    VAE = False
    if VAE:
        train_loss_tsdae = losses.VAELoss(model,
                                          latent_dim=256,
                                          decoder_name_or_path=PHASE1_DECODER,
                                          tie_encoder_decoder=True)
        print("VAE")
    else:
        train_loss_tsdae = losses.DenoisingAutoEncoderLoss(model,
                                                           decoder_name_or_path=PHASE1_DECODER,
                                                           tie_encoder_decoder=True)
    train_loss_sbert = losses.CosineSimilarityLoss(model)

    # ### CL
    evaluation_metric_sbert = "similarity"
    dataset_size = "extra_big"
    data_path_sbert = data_path_phase2
    scoring_ratio = 1  # This one does not affect for the regression case
    resize_data_syn = True if len(length.intersection({"medium", "long"})) else False
    resize_data_proportion = True
    eval_data = val_data_str
    ratio_max = 4
    classic = False

    # ### sbert dataloader
    sbert_data = pd.read_csv(data_path_sbert)
    sbert_data = sbert_data.dropna(how="any")

    train_dataloader_sbert, evaluator_sbert = sbert_data_preproces(
        sbert_data,
        resize_data_syn,
        resize_data_proportion,
        dataset_size,
        evaluation_metric_sbert,
        level,
        model_type,
        0.9,
        eval_data,
        scoring_ratio,
        ratio_max,
        classic,
        BATCH2,
        PHASE2_AUGMENTATION,
        *weights
    )

    # ### Union
    # This evaluator evaluates respect to both evaluation functions
    # but the optimization is only respect to the last one, the sbert one
    evaluators = [evaluator_tsdae, evaluator_sbert]
    seq_evaluator = evaluation.SequentialEvaluator(evaluators,
                                                   main_score_function=lambda scores:
                                                   scores[-1])

    model_name = 'MULTI_{}_{}_{}_{}'.format(dataset[0].split(".")[0].split("_")[-1].upper(),
                                            eval_string,
                                            basename(encoder_path),
                                            extra_name)
    output_name = "MULTI_" + "evaluated_" + os.path.basename(data_path_sbert).replace(".csv", "")
    final_multi = join(MULTI_PRETRAIN, output_name)

    # Check whether the specified path exists or not
    isExist = os.path.exists(final_multi)
    if not isExist:
        os.makedirs(final_multi)

    print("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=final_multi)

    steps_per_epoch = len(train_dataloader_tsdae) + len(train_dataloader_sbert)

    model.fit(
        train_objectives=[(train_dataloader_tsdae, train_loss_tsdae),
                          (train_dataloader_sbert, train_loss_sbert)],
        evaluator=None,
        epochs=EPOCHS_MULTI,
        scheduler="warmuplinear",
        warmup_steps=int((steps_per_epoch * EPOCHS_MULTI) / 2),
        optimizer_class=AdamW,
        optimizer_params={"lr": MULTI_LR},
        weight_decay=0,
        evaluation_steps=int((steps_per_epoch * EPOCHS_MULTI) / 5),  # / 10),
        output_path=final_multi,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True
    )

    # ############### PHASE 3 MULTITASK  ################
    print("\n############## PHASE 3 MULTITASK  ##############")
    torch.cuda.empty_cache()

    LR = PHASE3_LR
    eps = PHASE3_eps
    model_type = model_type
    device = check_device()

    base_config_json["base_model_path"] = final_multi
    base_config_json["embedding_size"] = base_model_config.hidden_size
    base_config_json["transformer"] = False
    base_config = BaseConfig(**base_config_json)

    sbert_data = read_csv(data_path_phase2)
    sbert_data = sbert_data.dropna(how="any")
    sbert_data = encoding_classification(sbert_data, hierarchy_level=level)

    # Create model if regression
    if "reg" in model_type:
        model = Classifier_head(base_config, num_class=1, architecture="complex")
        model.to(device)

    # Create model if classification
    elif "class" in model_type:
        model = Classifier_head(base_config, num_class=len(sbert_data["cat1"].unique()),
                                architecture="complex")
        model.to(device)

    train_dataloader_mt, test_dataloader = create_val_test_train(test_data_str, sbert_data, model_type, BATCH3)

    # Hyperparameters 2
    scheduler, optimizer = optimization_setup(train_dataloader_mt, model, LR, eps, PHASE3_EPOCHS_MAX)
    # Training
    train_loss, metric_dict_jl = trainer(PHASE3_EPOCHS_MIN, PHASE3_EPOCHS_MAX, train_dataloader_mt,
                                         test_dataloader, model,
                                         optimizer, device, scheduler, model_type, average)

    final_3_only = join(PHASE3_MULTI_PRETRAIN, "PHASE3_" + model_name)
    try:
        mkdir(final_3_only)
    except OSError as error:
        print(error)
    saving_path_bin = join(final_3_only, "model.bin")
    check_dir(saving_path_bin)
    torch.save(model.state_dict(), saving_path_bin)

    return metric_dict_jl


def check_dtype(df):
    cols = df.columns
    for col in cols:
        if "cat" in col:
            df[col] = df[col].astype('int')
        if "query" in col:
            df[col] = df[col].astype('str')
    return df


def ThreePhases(encoder, dataset, encoder_path, HYPER,
                list_extra_names=[], unbalance="normal",
                test_mode=False, long_mode=False, extra_text=""):
    """ One of the iterations in the loop for one of combination of hyperparameters.
    Input:
        - encoder (tuple): (models path: str, model type: str, lengths: list[str])
        - dataset (tuple):
                (path to dataset 1, path to dataset 2, path to validation dataset, path to test dataset,
                task type: "classification",lengths: list[str], level: if > 1 then hierarchical, metric,
                default MAX_LEN)
        - encoder_path (tuple):
        - HYPER (list): list of hyperparameters just unpackage below
    """
    (EPOCHS_PHASE1, EPOCHS_PHASE2, PHASE3_EPOCHS_MAX,
     PHASE1_LR, PHASE2_LR, PHASE3_LR,
     PHASE2_AUGMENTATION, PHASE3_eps,
     PHASE1_BATCH, PHASE2_BATCH, PHASE3_BATCH, MULTI_BATCH,
     MAX_LEN, PHASE1_DECODER, use_max_len, freezing, del_ratio,
     doexplode, given_max_len) = HYPER
    if not set(encoder[2]).intersection(set(dataset[5])):
        return None
    else:
        print(f'Encoder length {encoder[2]}. Dataset length {dataset[5]}')
        length = set(encoder[2]).intersection(set(dataset[5]))

    # ####################################################################################
    # #
    # # ####################################  PHASE 0  ####################################
    # #
    # ####################################################################################

    print("\n##############  PHASE 0  ##############")
    metric_dict_threephases = None

    logthem(encoder, ["model name", "type of model", "size of model"])
    logthem(dataset, ["path_phase1", "path_phase2", "val_path", "test_path", "ML_type", "size",
                      'level', "average", "max_len"])

    # Data paths: first to a series with the strings and second to a dataframe
    # with the categories or values of the regression
    data_path_1 = dataset[0]
    data_path_2 = dataset[1]
    val_data_str = dataset[2]
    test_data_str = dataset[3]
    average = dataset[7]

    # Do we use validation to choose the best model? If yes, which type of evaluation from the library sbert
    evaluate = True
    evaluation_metric = "bin" if average == "binary" else "sim"

    # Type of base model's architecture
    base_architecture_type = encoder[1]

    # level of the hierarchy
    level = dataset[6]

    # If the model is for regression or for classification
    model_type = dataset[4]

    base_model_config = AutoConfig.from_pretrained(encoder_path)

    # # explode data
    sbert_data = read_csv(data_path_2)
    df_val = read_csv(val_data_str).dropna() if val_data_str else None
    df_test = read_csv(test_data_str).dropna() if test_data_str else None
    sbert_data = sbert_data.dropna()
    sbert_data = check_dtype(sbert_data)
    if val_data_str:
        df_val = check_dtype(df_val)
    if test_data_str:
        df_test = check_dtype(df_test)
    if long_mode:
        # Adapt data to the model
        # max_len_original = base_model_config.max_position_embeddings - 10
        print(f"-------------- max_len_original {MAX_LEN} ----------")
        tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        if doexplode:
            df1, df2 = chunk_it(sbert_data, tokenizer, MAX_LEN)
            _, df_test = chunk_it(df_test, tokenizer, MAX_LEN)
        else:
            sbert_data["query"] = sbert_data["query"].apply(lambda x: text_split(x, tokenizer, MAX_LEN)[0])
            df1, df2 = sbert_data["query"].copy(), sbert_data.copy()
        if df_val is not None: df_val["query"] = df_val["query"].apply(lambda x: text_split(x, tokenizer, MAX_LEN)[0])
        if df_test is not None: df_test["query"] = df_test["query"].apply(
            lambda x: text_split(x, tokenizer, MAX_LEN)[0])
    else:
        df1 = read_csv(data_path_1)
        df2 = sbert_data.copy()
    data_path_phase1 = join("Data", "temp_df1.csv")
    data_path_phase2 = join("Data", "temp_df2.csv")
    if df_val is not None: val_data_str = join("Data", "temp_val.csv")
    if df_val is not None: test_data_str = join("Data", "temp_test.csv")
    if test_mode:
        df1 = df1[:200]
        df2 = df2[:200]
    if test_data_str is not None: df_test.to_csv(test_data_str, index=False)
    if val_data_str is not None: df_val.to_csv(val_data_str, index=False)
    df1.to_csv(data_path_phase1, index=False)
    df2.to_csv(data_path_phase2, index=False)

    R = 1 if df1.shape[0] > 500 else 1 if test_mode else 0.2
    # R = 1
    BATCH1 = int(PHASE1_BATCH * R)
    BATCH2 = int(PHASE2_BATCH * R)
    BATCH3 = int(PHASE3_BATCH * R)

    # Print the hyperparameters
    list1 = [MAX_LEN, BATCH1, EPOCHS_PHASE1, PHASE1_LR, PHASE1_ENCODER, PHASE1_DECODER,
             BATCH2, EPOCHS_PHASE2, PHASE2_AUGMENTATION, PHASE2_LR, PHASE2_MODEL,
             BATCH3, PHASE3_LR, PHASE3_eps, PHASE3_EPOCHS_MAX, freezing, del_ratio, given_max_len]
    list2 = ["MAX_LEN", "PHASE1_BATCH", "EPOCHS_PHASE1", "PHASE1_LR", "PHASE1_ENCODER", "PHASE1_DECODER",
             "PHASE2_BATCH", "EPOCHS_PHASE2", "PHASE2_AUGMENTATION", "PHASE2_LR",
             "PHASE2_MODEL", "PHASE3_BATCH", "PHASE3_LR", "PHASE3_eps",
             "PHASE3_EPOCHS_MAX", "freezing", "del_ratio", "given_max_len"]
    parameters = dict(zip(list2, list1))
    for k, v in parameters.items():
        print(k, str(v))

    # Shared name for all the phases for this model
    eval_string = "eval" if evaluate else ""
    extra_name = "_LR" + str(PHASE1_LR) + str(PHASE2_LR) + str(PHASE3_LR) \
                 + "_EP" + str(EPOCHS_PHASE1) + str(EPOCHS_PHASE2) + str(PHASE3_EPOCHS_MAX) \
                 + "_BA" + str(PHASE1_BATCH) + str(PHASE2_BATCH) + str(PHASE3_BATCH) \
                 + "_eps" + str(PHASE3_eps) + "AUG" + str(PHASE2_AUGMENTATION) \
                 + "U" + str(1 if use_max_len else 0) + "G" + str(1 if given_max_len else 0) \
                 + "F" + str(1 if freezing else 0) + "D" + str(del_ratio) + "E" + str(1 if doexplode else 0)
    list_extra_names.append(extra_name)
    model_name = '{}_{}_{}_{}'.format(dataset[0].split(".")[0].split("_")[-1].upper(),
                                      eval_string,
                                      basename(encoder_path).replace("sentence-transformers", ""),
                                      extra_name)

    # ###################################################################################
    #
    # # ####################################  PHASE 1  ####################################
    #
    # ####################################################################################
    print("\n##############  PHASE 1  ##############")
    torch.cuda.empty_cache()
    # print("\n DATA LOADED {}\n".format(df2.head(2)))

    # ############ Training phase 1
    final = join(PHASE1_TRAIN, "PHASE1_" + extra_text + model_name)
    final = dae_trainer(final,
                        data_path_phase1,
                        data_path_phase2,
                        evaluate,
                        model_type,
                        evaluation_metric,
                        use_max_len=use_max_len,
                        max_len=MAX_LEN,
                        base_type=base_architecture_type,
                        encoder=encoder_path,
                        level=level,
                        VAE=False,
                        epochs=EPOCHS_PHASE1,
                        BATCH=BATCH1,
                        LR=PHASE1_LR,
                        del_ratio=del_ratio)
    print("Phase 1 checkpoint path", final)

    # ####################################################################################
    #
    # # ####################################  PHASE 2  ###################################
    #
    # ####################################################################################

    print("\n##############  PHASE 2  ##############")
    torch.cuda.empty_cache()

    # Variables
    evaluation_metric = "similarity"
    model_path = final
    dataset_size = "extra_big"
    base_architecture_type = "stransformer"
    scoring_ratio = 1
    resize_data_syn = True if len(length.intersection({"medium", "long"})) else False
    resize_data_proportion = True
    eval_data = val_data_str
    if unbalance == "extra":
        ratio_max = 20
        ratio_min = 1.5
    elif unbalance == "normal":
        ratio_max = 4
        ratio_min = 1.5
    else:
        ratio_max = 1
        ratio_min = 1
    classic = False
    train_portion = 0.9
    evaluate = True
    weights = [1]
    pooling_mode = 'cls'
    encoder_name = encoder_path
    print(f" data_path_phase2  {data_path_phase2}")
    print(f" eval_data  {eval_data}")

    final_2 = training_sbert(join(PHASE2_TRAIN, "PHASE2_" + extra_text + model_name),
                             evaluation_metric,
                             level,
                             model_type,
                             model_path,
                             dataset_size,
                             base_architecture_type,
                             resize_data_syn,
                             resize_data_proportion,
                             data_path_phase2,
                             eval_data,
                             scoring_ratio,
                             ratio_max,
                             classic,
                             pooling_mode,
                             train_portion,
                             evaluate,
                             encoder_name,
                             EPOCHS_PHASE2,
                             PHASE2_LR,
                             PHASE2_AUGMENTATION,
                             BATCH2,
                             ratio_min,
                             *weights
                             )
    print("phase 2", final_2)

    # ####################################################################################
    #
    # ###################################  PHASE 3  ######################################
    #
    # ####################################################################################

    print("\n##############  PHASE 3  ##############")
    torch.cuda.empty_cache()
    # Configuration
    device = check_device()
    truncation = 512 if length == "medium" else 128 if length == "short" else 1024
    base_config_json = {
        "freezing": freezing,
        "device": device,
        "base_model_path": final_2,
        "embedding_size": base_model_config.hidden_size,
        "transformer": False,
        "model_type": model_type,
        "truncation": truncation,
        "from_embeddings": False
    }
    base_config = BaseConfig(**base_config_json)

    # Data preprocess
    sbert_data = pd.read_csv(data_path_phase2)
    sbert_data = sbert_data.dropna(how="any")
    sbert_data = encoding_classification(sbert_data, hierarchy_level=level)

    # Create model if regression
    if "reg" in model_type:
        model = Classifier_head(base_config, num_class=1, architecture="complex")
        model.to(device)

    # Create model if classification
    elif "class" in model_type:
        model = Classifier_head(base_config, num_class=len(sbert_data["cat1"].unique()),
                                architecture="complex")
        model.to(device)
    else:
        print("ERROR on the task type")
        model = None

    train_dataloader, test_dataloader = create_val_test_train(test_data_str, sbert_data, model_type, BATCH3)

    # Hyperparameters 2
    scheduler, optimizer = optimization_setup(train_dataloader, model, PHASE3_LR, PHASE3_eps, PHASE3_EPOCHS_MAX)

    train_loss, metric_dict_threephases = trainer(PHASE3_EPOCHS_MIN, PHASE3_EPOCHS_MAX, train_dataloader,
                                           test_dataloader, model, optimizer, device, scheduler,
                                           model_type, average)

    final_3 = join(PHASE3_TRAIN, "PHASE3_" + extra_text + model_name)
    print("phase 3 checkpoint path", final_3)
    try:
        mkdir(final_3)
    except OSError as error:
        print(error)
    saving_path_bin = join(final_3, "model.bin")
    check_dir(saving_path_bin)
    torch.save(model.state_dict(), saving_path_bin)

    save_name = '{}_{}_{}'.format(dataset[0].split(".")[0].split("_")[-1].upper(),
                                  basename(encoder_path),
                                  extra_name)
    save_and_plot(metric_dict_threephases, "main" + extra_text, save_name)

    for dataset in ["SST2"]:
        for extra in list_extra_names:
            onlyfiles = [f for f in listdir(RESULTS) if isfile(join(RESULTS, f))]
            list_pkls = [x for x in onlyfiles if dataset in x]
            list_pkls = [x for x in list_pkls if extra in x]
            plot_comparison(list_pkls, ignore=["confusion"], dataset=dataset, save_name=save_name)
