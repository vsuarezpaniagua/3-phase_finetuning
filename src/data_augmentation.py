import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import math
import logging

from src.config import APPSTOP

logger = logging.getLogger(__name__)


def score_balancer(df, counts, threshold, mode):
    """ This function balance the proportion of samples based on the score.
        Counts must be the value_counts of the score.
        It takes at most n=threshold examples with the same score """
    logger.debug("{} counts before balance:\n{}\n".format(mode, counts))
    threshold = int(threshold)
    matches = counts.index.tolist()
    evaluation_metric = [df[df["score"] == match].sample(n=threshold)
                         if counts[match] >= threshold else df[df["score"] == match].sample(
        n=counts[match])
                         for match in matches]
    df = pd.concat(evaluation_metric).reset_index(drop=True)
    logger.debug("{} counts after balance:\n{}\n".format(mode, df["score"].value_counts()))
    return shuffle(df)


def resize_join_synthetic(df_list, maximum, classic=False, ratio_min=2, ratio_max=20):
    """ This function joins a list of dataframes (df_list), in a balanced manner based on ratio """
    nope = True if ratio_min == ratio_min and ratio_min == 1 else False
    appstopw = stop_words(nope=nope)
    query_col = [col for col in df_list[-1].columns.tolist() if "cat" not in col][0]
    if ratio_min == ratio_min and ratio_min == 1:
        function = lambda x: 1
        print("#########   NO RESIZING")
    else:
        function = lambda x: math.log(maximum / x) * (ratio_max / math.log(maximum))

    total_list = []
    pbar = tqdm(df_list)
    for df in pbar:
        pbar.set_description("Resizing artificial ...")
        proportion = int(maximum / df.shape[0])

        if classic:
            if proportion > ratio_min:
                total_list.append(expander(df, query_col, appstopw))
            else:
                total_list.append(df)

        # The dataframes are resized but with a maximum
        else:
            my_ration = function(df.shape[0])
            if my_ration >= ratio_min:
                total_list.append(expander(df, query_col, appstopw))
            else:
                total_list.append(df)

    return pd.concat(total_list, ignore_index=True, axis=0)


def resize_join_multiplying(df_list, maximum, classic, ratio_min=1.5, ratio_max=20):
    """ This function joins a list of dataframes (df_list) in a balanced manner based on a ratio.
    Input:
        - df_list (list(pd.Dataframe)):
        - maximum (int): biggest size in the list of dataframes
        - classic (Bool): If False a dataframe can not be multiplied more than
        - ratio_min (float): minimum ration to resize.
            If the the dataframe is ratio_min times smaller it is not resized
        - ratio_max (float): maximum ration to resize
    """
    final_list = []
    pbar = tqdm(df_list)
    if ratio_min == ratio_max == 1:
        function = lambda x: 1
        print("#########   NO RESIZING")
    else:
        function = lambda x: math.log(maximum / x) * (ratio_max / math.log(maximum))

    for df in pbar:
        pbar.set_description(
            "Resizing the data based on the labels in the {} way ...".format("classic" if classic else "NOT classic"))
        proportion = int(maximum / df.shape[0])

        # All dataframes are going to be resized independently of their size
        if classic:
            if proportion >= ratio_min:
                final_list.append(pd.concat([df] * proportion, ignore_index=True))
            else:
                final_list.append(df)

        # The dataframes are resized but with a maximum
        else:
            my_ration = function(df.shape[0])
            if my_ration >= ratio_min:
                final_list.append(pd.concat([df] * round(my_ration), ignore_index=True))
            else:
                final_list.append(df)
    return pd.concat(final_list, ignore_index=True, axis=0)


def expander(df, col, appstopw):
    # TODO: this function should sample the masked stopwords not just replace all of them
    """ This function increase the number of samples (rows) by deleting the stopwords
    and not useful names from the queries while adding them as new data. It is used to increase data
    on those underrepresented classes not just multiplying the data"""
    mylist = df[col].tolist()
    columns = df.columns.tolist()
    mylist = [w.lower() for w in mylist]
    new_queries = []
    for query in mylist:
        new = False
        new_query = []
        query = query.split()
        for word in query:
            if word in appstopw:
                new = True
            else:
                new_query.append(word)
        if new:
            if len(new_query) >= 1:
                new_queries.append(" ".join(new_query))
            new_queries.append(" ".join(query))
        else:
            new_queries.append(" ".join(query))
    df1 = pd.DataFrame(columns=columns)
    df1[col] = new_queries
    columns.remove(col)
    for column in columns:
        value = df[column].unique().tolist()[0]
        df1[column] = [value] * len(new_queries)
    return df1.reset_index(drop=True)


def stop_words(nope=False):
    """ It loads the stopwords for each given CC . If multi=True then the 5 """
    if not nope:
        with open(APPSTOP, "r") as file:
            stopw = file.read()
            file.close()
        appstopw = [word.strip() for word in stopw.split("\n")]
    else:
        appstopw = []
    return set(appstopw)


def get_maximums(df, level_total, level_current):
    """ This function computes the balance of the dataset.
    Input:
        - df (pd.Dataframe):
        - level_total (int):
        - level_current (int):
    Output:
        - maxn (int): size of the biggest category of level n
        - df_2_all (list(DataFrames)): this is df grouped by category level 2
    """
    if level_total == 1:
        maxn, df_list = get_maximums1(df)
    elif level_total == 2:
        _, maxn, df_list = get_maximums2(df)
    elif level_total == 3:
        _, _, maxn, df_list = get_maximums3(df)
    return maxn, df_list


# def get_maximums(df, level_total, level_current, parent_cat=None, maxn=1):
#     """ This function computes the balance of the dataset.
#     Input:
#         - df (pd.Dataframe):
#         - level_total (int):
#         - level_current (int):
#         - parent_cat (str):
#         - maxn (int):
#     Output:
#         - maxn (int): size of the biggest category of level n
#         - df_2_all (list(DataFrames)): this is df grouped by category level 2
#     """
#     level_current += 1
#     this_cat = "cat" + str(level_current)
#     next_cat = "cat" + str(level_current + 1)
#     previous_cat = "cat" + str(level_current + 1)
#     cat_list = []
#     df_list = []
#
#     for cat, count in list(zip(df.index, df)):
#         cat_list.append(cat)
#         if level_total > level_current:
#             df_previous = df[df[this_cat] == cat][next_cat].value_counts()
#             count, df_list_nex = get_maximums(df_previous, level_total, level_current + 1, cat)
#             df_list.extend(df_list_nex)
#         else:
#             if parent_cat:
#                 df_list.append(df[(df[this_cat] == cat) & (df[previous_cat] == parent_cat)])
#             else:
#                 df_list.append(df[df[this_cat] == cat])
#         maxn = count if count > maxn else maxn
#     return maxn, df_list


def get_maximums1(df):
    """ This function computes the balance of the dataset.
    :return
        - max1 (int): size of the biggest category of level 1
        - df_all (list(DataFrames)): this is df grouped by category level 1
    """
    cat1_list = []
    max1 = 0
    df_all = []
    p = df["cat1"].value_counts()
    for cat1, count1 in list(zip(p.index, p)):
        cat1_list.append(cat1)
        if count1 > max1:
            max1 = count1
        else:
            pass
        df_all.append(df[df["cat1"] == cat1])
    return max1, df_all


def get_maximums2(df):
    """ This function computes the balance of the dataset.
    :return
        - max1 (int): size of the biggest category of level 1
        - max2 (int): size of the biggest category of level 2
        - df_2_all (list(DataFrames)): this is df grouped by category level 2
    """
    cat1_list = []
    max1 = 0
    cat2_list = []
    max2 = 0
    df_2_all = []
    p = df["cat1"].value_counts()
    for cat1, count1 in list(zip(p.index, p)):
        cat1_list.append(cat1)
        if count1 > max1:
            max1 = count1
        else:
            pass
        p2 = df[df["cat1"] == cat1]["cat2"].value_counts()
        for cat2, count2 in list(zip(p2.index, p2)):
            cat2_list.append(cat2)
            if count2 > max2:
                max2 = count2
            else:
                pass
            df_2_all.append(df[(df["cat1"] == cat1) & (df["cat2"] == cat2)])
    return max1, max2, df_2_all


def get_maximums3(df):
    """ This function computes the balance of the dataset.
    :return
        - max1 (int): size of the biggest category of level 1
        - max2 (int): size of the biggest category of level 2
        - df_2_all (list(DataFrames)): this is df grouped by category level 2
    """
    cat1_list, cat2_list, cat3_list, df_3_all = [], [], [], []
    max1, max2, max3 = 0, 0, 0
    p = df["cat1"].value_counts()
    for cat1, count1 in list(zip(p.index, p)):
        cat1_list.append(cat1)
        if count1 > max1:
            max1 = count1
        else:
            pass
        p2 = df[df["cat1"] == cat1]["cat2"].value_counts()
        for cat2, count2 in list(zip(p2.index, p2)):
            cat2_list.append(cat2)
            if count2 > max2:
                max2 = count2
            else:
                pass
            p3 = df[(df["cat1"] == cat1) & (df["cat2"] == cat2)]["cat3"].value_counts()
            for cat3, count3 in list(zip(p3.index, p3)):
                cat3_list.append(cat2 + cat3)
                if count3 > max3:
                    max3 = count3
                else:
                    pass
                df_3_all.append(df[(df["cat3"] == cat3) & (df["cat2"] == cat2)])
    return max1, max2, max3, df_3_all
