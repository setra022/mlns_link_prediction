import numpy as np
import pandas as pd
import scipy
import csv
import json
import pickle
import os
import nltk
import igraph
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

node_info = pd.read_csv("node_info.csv", header=None)
node_info.columns = ['id', 'year', 'title', 'authors', 'journal_name', 'abstract']
node_info.fillna('', inplace=True)
node_info.set_index('id', inplace=True)

print("Loading training set...")
training_set = pd.read_csv("training_set.txt", sep=' ', header=None)
training_set.columns = ['source', 'target', 'edge']

print("Loading testing set...")
testing_set = pd.read_csv("testing_set.txt", sep=' ', header=None)
testing_set.columns = ['source', 'target']

print("Loading text features...")
with open('./resource/text_features.json') as json_file:
    text_features_dic = json.load(json_file)

print("Vectorizing corpus...")
corpus = node_info.abstract
vectorizer = TfidfVectorizer(stop_words="english")
features_TFIDF = vectorizer.fit_transform(corpus)

def get_stems(title):
    title = title.lower()
    words = set(title.split(' ')) - stpwds
    words = set(map(lambda x: stemmer.stem(x), words))
    return words

def get_authors(authors):
    authors = map(lambda s: s.strip(), authors.split(','))
    authors = set(filter(lambda s: s != '', authors))
    return authors

def find_node(graph, nodes_index, name):
    return graph.vs[nodes_index[name]]

def get_features(row, nodes_index, graph, katz):
    source_node = find_node(graph, nodes_index, row.source)
    target_node = find_node(graph, nodes_index, row.target)

    pref_attachment = source_node.degree() * target_node.degree()

    u1 = features_TFIDF[source_node.index]
    u2 = features_TFIDF[target_node.index]

    corr = np.dot(u1, u2.T).toarray()[0, 0]

    katz_measure = katz[source_node.index, target_node.index]

    # len_shortest_path = len(shortest_paths[0])
    # import pdb;pdb.set_trace()
    # source_neighbors_in = set(graph.neighbors(source_node, mode='in'))
    # source_neighbors_out = set(graph.neighbors(source_node, mode='out'))

    # target_neighbors_in = set(graph.neighbors(target_node, mode='in'))
    # target_neighbors_out = set(graph.neighbors(target_node, mode='out'))

    # common_in = len(source_neighbors_in.intersection(target_neighbors_in))
    # common_out = len(source_neighbors_out.intersection(target_neighbors_out))
    source_neighbors = set(graph.neighbors(source_node))
    target_neighbors = set(graph.neighbors(target_node))

    common = source_neighbors.intersection(target_neighbors)
    union = source_neighbors.union(target_neighbors)
    jaccard = len(common) / len(union) if len(union) > 0 else 0
    cosine = len(common) / pref_attachment if pref_attachment > 0 else 0

    adamic = sum([1 / np.log(graph.vs[node].degree()) for node in common])
    source_info = node_info.loc[row.source]
    target_info = node_info.loc[row.target]

    source_title_stems = get_stems(source_info.title)
    target_title_stems = get_stems(target_info.title)
    overlap_title = len(source_title_stems.intersection(target_title_stems))

    temp_diff = source_info.year - target_info.year

    source_authors = get_authors(source_info.authors)
    target_authors = get_authors(target_info.authors)
    comm_auth = len(source_authors.intersection(target_authors))
    
    text_features = text_features_dic[str(row.source)][str(row.target)]

    return [overlap_title, temp_diff, comm_auth, jaccard, adamic, corr, pref_attachment, cosine, katz_measure] + text_features

def get_katz_matrix(graph, beta, depth):
    print("Computing Katz matrix...")
    adj = graph.get_adjacency_sparse()
    # adj = (adj + adj.transpose()) / 2
    b_adj = beta * adj
    product = b_adj.dot(b_adj)
    katz = scipy.sparse.csr_matrix((len(node_info), len(node_info)))
    for _ in range(depth):
        katz += product
        product = product.dot(b_adj)
    return katz

def load_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)

def save_data(data_path, data):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def add_features(X_train, X_test):
    graph = igraph.Graph(directed=True)
    nodes = list(node_info.index)
    graph.add_vertices(nodes)

    nodes_index = {}

    for node in graph.vs:
        nodes_index[node["name"]] = node.index

    edges = []
    for index, row in X_train.iterrows():
        if y_train[index] == 1:
            source = nodes_index[row.source]
            target = nodes_index[row.target]
            edges.append((source, target))

    graph.add_edges(edges)

    katz = get_katz_matrix(graph, 0.5, 5)

    training_features = []
    for _, row in tqdm(list(X_train.iterrows()), desc="Computing training features..."):
        training_features.append(get_features(row, nodes_index, graph, katz))

    X_train = np.array(training_features)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    val_features = []
    for _, row in tqdm(list(X_test.iterrows()), desc="Computing validation features..."):
        val_features.append(get_features(row, nodes_index, graph, katz))

    X_test = np.array(val_features)

    X_test = scaler.transform(X_test)

    return X_train, X_test

def run(X_train, X_test, y_train, data_path=None):

    #model = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')
    #model = LinearSVC()
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("Model fitting...")
    model.fit(X_train, y_train)
    # import pdb;pdb.set_trace()
    print("Fitting done.")

    y_pred = model.predict(X_test)

    return y_pred


validate_model = False
data_path = None

if validate_model:

    # N_samples = 10000
    # training_set = training_set[:N_samples]

    X = training_set[['source', 'target']]
    y = training_set['edge']
    if data_path is not None and os.path.exists(data_path):
        X_train, X_val, y_train, y_val = load_data(data_path)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        X_train, X_val = add_features(X_train, X_val)
        filepath = data_path or "./data.pickle"
        save_data(filepath, (X_train, X_val, y_train, y_val))

    y_pred = run(X_train, X_val, y_train, data_path=data_path)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision : {precision:.4f}')
    print(f'Recall : {recall:.4f}')
    print(f'F1-Score : {f1:.4f}')

    print("Done.")

else:
    X_train = training_set[['source', 'target']]
    X_test = testing_set
    y_train = training_set['edge']

    X_train, X_test = add_features(X_train, X_test)

    y_pred = run(X_train, X_test, y_train)

    y_pred = zip(range(len(testing_set)), y_pred)

    with open("test_predictions.csv", "w") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(["id", "category"])
        for row in y_pred:
            csv_out.writerow(row)
