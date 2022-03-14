import numpy as np
import pandas as pd
import csv
import nltk
import igraph
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein as lev
from fuzzywuzzy import fuzz

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

node_info = pd.read_csv("node_info.csv", header=None)
node_info.columns = ['id', 'year', 'title', 'authors', 'journal_name', 'abstract']
node_info.fillna('', inplace=True)
node_info.set_index('id', inplace=True)

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

def get_features(row, nodes_index, graph):
    source_node = find_node(graph, nodes_index, row.source)
    target_node = find_node(graph, nodes_index, row.target)

    pref_attachment = source_node.degree() * target_node.degree()

    u1 = features_TFIDF[source_node.index]
    u2 = features_TFIDF[target_node.index]

    corr = np.dot(u1, u2.T).toarray()[0, 0]

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
    
    
    lev_title_dist = lev.distance(source_info.title,target_info.title)
    lev_title_ratio = lev.ratio(source_info.title,target_info.title)
    lev_desc_dist = lev.distance(source_info.abstract,target_info.abstract)
    lev_desc_ratio = lev.ratio(source_info.abstract,target_info.abstract)
    lev_journal_dist = lev.distance(source_info.journal_name,target_info.journal_name)
    lev_journal_ratio = lev.ratio(source_info.journal_name,target_info.journal_name)
    fuzz_ratio_t = fuzz.ratio(source_info.title,target_info.title)
    fuzz_partial_ratio_t = fuzz.partial_ratio(source_info.title,target_info.title)
    fuzz_token_sort_ratio_t = fuzz.token_sort_ratio(source_info.title,target_info.title)
    fuzz_token_set_ratio_t = fuzz.token_set_ratio(source_info.title,target_info.title)
    fuzz_ratio_desc = fuzz.ratio(source_info.abstract,target_info.abstract)
    fuzz_partial_ratio_desc = fuzz.partial_ratio(source_info.abstract,target_info.abstract)
    fuzz_token_sort_ratio_desc = fuzz.token_sort_ratio(source_info.abstract,target_info.abstract)
    fuzz_token_set_ratio_desc = fuzz.token_set_ratio(source_info.abstract,target_info.abstract)
    fuzz_ratio_j = fuzz.ratio(source_info.journal_name,target_info.journal_name)
    fuzz_partial_ratio_j = fuzz.partial_ratio(source_info.journal_name,target_info.journal_name)
    fuzz_token_sort_ratio_j = fuzz.token_sort_ratio(source_info.journal_name,target_info.journal_name)
    fuzz_token_set_ratio_j = fuzz.token_set_ratio(source_info.journal_name,target_info.journal_name)

    return [overlap_title, temp_diff, comm_auth, jaccard, adamic, corr, pref_attachment, lev_title_dist, lev_title_ratio, lev_desc_dist, lev_desc_ratio,lev_journal_dist,lev_journal_ratio,fuzz_ratio_t,fuzz_partial_ratio_t,fuzz_token_sort_ratio_t,fuzz_token_set_ratio_t,fuzz_ratio_desc,fuzz_partial_ratio_desc,fuzz_token_sort_ratio_desc,fuzz_token_set_ratio_desc,fuzz_ratio_j,fuzz_partial_ratio_j,fuzz_token_sort_ratio_j,fuzz_token_set_ratio_j]


def run(X_train, X_test, y_train):
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

    training_features = []
    for _, row in tqdm(list(X_train.iterrows()), desc="Computing training features..."):
        training_features.append(get_features(row, nodes_index, graph))

    training_features = np.array(training_features)
    cols = ['overlap_title', 'temp_diff', 'comm_auth', 'jaccard', 'adamic', 'corr', 'pref_attachment','lev_title_dist', 'lev_title_ratio', 'lev_desc_dist', 'lev_desc_ratio','lev_journal_dist','lev_journal_ratio','fuzz_ratio_t','fuzz_partial_ratio_t','fuzz_token_sort_ratio_t','fuzz_token_set_ratio_t','fuzz_ratio_desc','fuzz_partial_ratio_desc','fuzz_token_sort_ratio_desc','fuzz_token_set_ratio_desc','fuzz_ratio_j','fuzz_partial_ratio_j','fuzz_token_sort_ratio_j','fuzz_token_set_ratio_j']

    X_train[cols] = training_features
    X_train = X_train[cols]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    val_features = []
    for _, row in tqdm(list(X_test.iterrows()), desc="Computing validation features..."):
        val_features.append(get_features(row, nodes_index, graph))

    val_features = np.array(val_features)

    X_test[cols] = val_features
    X_test = X_test[cols]

    X_test = scaler.transform(X_test)

    model=RandomForestClassifier(n_estimators = 300, class_weight = 'balanced')
    #model = LinearSVC()
    print("Model fitting...")
    model.fit(X_train, y_train)
    print("Fitting done.")

    y_pred = model.predict(X_test)

    return y_pred


training_set = pd.read_csv("training_set.txt", sep=' ', header=None)
training_set.columns = ['source', 'target', 'edge']

testing_set = pd.read_csv("testing_set.txt", sep=' ', header=None)
testing_set.columns = ['source', 'target']


validate_model = False

if validate_model:

    # N_samples = 100000
    # training_set = training_set[:N_samples]

    X = training_set[['source', 'target']]
    y = training_set['edge']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    y_pred = run(X_train, X_val, y_train)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f'Accuracy : {accuracy:.2f}')
    print(f'Precision : {precision:.2f}')
    print(f'Recall : {recall:.2f}')
    print(f'F1-Score : {f1:.2f}')
    import pdb;pdb.set_trace()

    print("Done.")

else:
    X = training_set[['source', 'target']]
    y = training_set['edge']

    y_pred = run(X, testing_set, y)

    y_pred = zip(range(len(testing_set)), y_pred)

    with open("test_predictions.csv", "w") as pred1:
        csv_out = csv.writer(pred1)
        for row in y_pred:
            csv_out.writerow(row)
