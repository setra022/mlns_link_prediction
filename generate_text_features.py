import pandas as pd
import json
import Levenshtein as lev
from fuzzywuzzy import fuzz
from tqdm import tqdm
import os

node_info = pd.read_csv("node_info.csv", header=None)
node_info.columns = ['id', 'year', 'title', 'authors', 'journal_name', 'abstract']
node_info.fillna('', inplace=True)
node_info.set_index('id', inplace=True)

training_set = pd.read_csv("training_set.txt", sep=' ', header=None)
training_set.columns = ['source', 'target', 'edge']

testing_set = pd.read_csv("testing_set.txt", sep=' ', header=None)
testing_set.columns = ['source', 'target']

# training_set = training_set[:10]
# testing_set = training_set[:10]

text_features_dic = {}

def get_text_features(source_id, target_id):

    source_info = node_info.loc[source_id]
    target_info = node_info.loc[target_id]

    lev_title_dist = lev.distance(source_info.title, target_info.title)
    lev_title_ratio = lev.ratio(source_info.title, target_info.title)

    lev_desc_dist = lev.distance(source_info.abstract, target_info.abstract)
    lev_desc_ratio = lev.ratio(source_info.abstract, target_info.abstract)

    lev_journal_dist = lev.distance(source_info.journal_name, target_info.journal_name)
    lev_journal_ratio = lev.ratio(source_info.journal_name, target_info.journal_name)

    fuzz_ratio_t = fuzz.ratio(source_info.title, target_info.title)
    fuzz_partial_ratio_t = fuzz.partial_ratio(source_info.title, target_info.title)
    fuzz_token_sort_ratio_t = fuzz.token_sort_ratio(source_info.title, target_info.title)
    fuzz_token_set_ratio_t = fuzz.token_set_ratio(source_info.title, target_info.title)

    fuzz_ratio_desc = fuzz.ratio(source_info.abstract, target_info.abstract)
    fuzz_partial_ratio_desc = fuzz.partial_ratio(source_info.abstract, target_info.abstract)
    fuzz_token_sort_ratio_desc = fuzz.token_sort_ratio(source_info.abstract, target_info.abstract)
    fuzz_token_set_ratio_desc = fuzz.token_set_ratio(source_info.abstract, target_info.abstract)

    fuzz_ratio_j = fuzz.ratio(source_info.journal_name, target_info.journal_name)
    fuzz_partial_ratio_j = fuzz.partial_ratio(source_info.journal_name, target_info.journal_name)
    fuzz_token_sort_ratio_j = fuzz.token_sort_ratio(source_info.journal_name, target_info.journal_name)
    fuzz_token_set_ratio_j = fuzz.token_set_ratio(source_info.journal_name, target_info.journal_name)

    return [lev_title_dist, lev_title_ratio, lev_desc_dist, lev_desc_ratio, lev_journal_dist, lev_journal_ratio,
                    fuzz_ratio_t, fuzz_partial_ratio_t, fuzz_token_sort_ratio_t, fuzz_token_set_ratio_t, fuzz_ratio_desc,
                    fuzz_partial_ratio_desc, fuzz_token_sort_ratio_desc, fuzz_token_set_ratio_desc,fuzz_ratio_j,
                    fuzz_partial_ratio_j, fuzz_token_sort_ratio_j, fuzz_token_set_ratio_j]


def add_row_info(row):
    source_id, target_id = row.source, row.target
    text_features = get_text_features(source_id, target_id)

    source_id, target_id = str(source_id), str(target_id)

    if source_id not in text_features_dic:
        text_features_dic[source_id] = {}

    text_features_dic[source_id][target_id] = text_features

for _, row in tqdm(list(training_set.iterrows()), desc="Computing text features of training set..."):
    add_row_info(row)

for _, row in tqdm(list(testing_set.iterrows()), desc="Computing text features of testing set..."):
    add_row_info(row)

if not os.path.exists("./resource/"):
    os.makedirs("./resource/")

with open('./resource/text_features.json', 'w') as outfile:
    json.dump(text_features_dic, outfile, indent=4)

print("Done.")