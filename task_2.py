import time
from os import chdir, listdir, rename
from os.path import dirname, join

working_dir = dirname(__file__)
chdir(working_dir)

from nltk import download, word_tokenize
download("punkt")

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

def preprocess_reviews_to_files(review_dir, pickle_dir):
    reviews = listdir(review_dir)
    for review in reviews:
        text = open(join(review_dir, review), 'r', encoding="utf-8").read()
        token_list = word_tokenize(text)
        token_list = [ps.stem(token) for token in token_list]
        doc = TaggedDocument(token_list, [review])
        pickle.dump(doc, open(join(pickle_dir, review) + ".p", "wb"))

def load_documents(pickle_dir):
    documents = listdir(pickle_dir)
    docs = []
    for document in documents:
        with open(join(pickle_dir, document), "rb") as f:    
            doc = pickle.load(f)
            docs.append(doc)
    return docs

def get_stratified_split(docs, num_folds, offset):
    train_reviews = []
    test_reviews = []

    i = 0
    while i < len(docs):
        test_reviews.append(docs[i + offset])
        for j in range(1, num_folds):
            index = i + ((j + offset) % num_folds)
            train_reviews.append(docs[index])
        i += num_folds

    return (train_reviews, test_reviews)

from collections import Counter

def make_dictionary(documents):
    all_tokens = []
    for document in documents:
        seen_tokens = []
        for token in document.words:
            if token not in seen_tokens:
                all_tokens.append(token)
                seen_tokens.append(token)
    dictionary = Counter(all_tokens)
    dictionary = dictionary.most_common(len(dictionary))
    dictionary = [(t, n) for (t, n) in dictionary if n > feature_cutoff]
    dictionary.sort(key = lambda tup: tup[0])
    return dictionary

from bisect import bisect_left

def extract_features(documents, dictionary):
    features_matrix = np.zeros((len(documents), len(dictionary)))
    docID = 0
    for document in documents:
        for token in document.words:
            tokenID = 0
            i = bisect_left(dictionary, (token, 0))
            if i != len(dictionary) and dictionary[i][0] == token:
                tokenID = i
                features_matrix[docID, tokenID] = 1
        docID = docID + 1
    return features_matrix

from random import shuffle

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

num_folds = 10
feature_cutoff = 7

#  THINGS TO CHANGE

vector_size = 100
epochs = 20
# /THINGS TO CHANGE

def main():
    start_time = time.process_time()

    #preprocess_reviews_to_files("data/neg", "data/neg_tokenized")
    #preprocess_reviews_to_files("data/pos", "data/pos_tokenized")
    #preprocess_reviews_to_files("data/unsup", "data/unsup_tokenized")

    print("Loading documents...")
    neg_docs = load_documents("data/neg_tokenized")
    pos_docs = load_documents("data/pos_tokenized")
    unsup_docs = load_documents("data/unsup_tokenized")

    '''
    (train_docs, test_docs) = get_stratified_split(neg_docs + pos_docs, num_folds, 0)
    
    print("Creating dictionary...")
    dictionary = make_dictionary(train_docs)

    print("Extracting training features...")
    train_labels_1 = np.zeros(len(train_docs))
    train_labels_1[len(train_docs)//2:] = 1
    train_matrix_1 = extract_features(train_docs, dictionary)

    print("Training models...")
    nb_1 = MultinomialNB()
    nb_1.fit(train_matrix_1, train_labels_1)

    svm_1 = SVC(kernel="linear")
    svm_1.fit(train_matrix_1, train_labels_1)

    print("Extracting test features...")
    test_labels_1 = np.zeros(len(test_docs))
    test_labels_1[len(test_docs)//2:] = 1
    test_matrix_1 = extract_features(test_docs, dictionary)

    print("Getting accuracy scores...")
    result_nb_1 = nb_1.predict(test_matrix_1)
    score_nb_1 = accuracy_score(test_labels_1, result_nb_1)
    result_svm_1 = svm_1.predict(test_matrix_1)
    score_svm_1 = accuracy_score(test_labels_1, result_svm_1)

    print("nb score is " + str(score_nb_1))
    print("svm score is " + str(score_svm_1))

    '''
    all_docs = neg_docs + pos_docs + unsup_docs

    doc2vec = Doc2Vec(min_count=1, window=10, vector_size=vector_size, sample=1e-4, negative=5, workers=8)
    doc2vec.build_vocab(all_docs)

    for epoch in range(epochs):
        shuffle(all_docs)
        doc2vec.train(all_docs, total_examples=doc2vec.corpus_count, epochs=1)

    print(doc2vec.most_similar("pay"))
    print(doc2vec.most_similar("extra"))

    '''
    train_matrix = np.zeros((len(train_docs), vector_size))
    for i, doc in enumerate(train_docs):
        train_matrix[i] = doc2vec.infer_vector(doc.words)
    
    train_labels = np.zeros(len(train_docs))
    train_labels[len(train_docs)//2:] = 1

    model = SVC(kernel="linear")
    model.fit(train_matrix, train_labels)

    test_matrix = np.zeros((len(test_docs), vector_size))
    for i, doc in enumerate(test_docs):
        test_matrix[i] = doc2vec.infer_vector(doc.words)
    test_labels = np.zeros(len(test_docs))
    test_labels[len(test_docs)//2:] = 1

    result = model.predict(test_matrix)
    score = accuracy_score(test_labels, result)

    print("accuracy score is " + str(score))
    '''

    print("running time is " + str(time.process_time() - start_time))

main()

def rename_files(file_dir, output_dir):
    files = listdir(file_dir)
    for i, f in enumerate(files):
        name = str(12500 + i) + "_" + f.split("_")[1]
        rename(join(file_dir, f), join(output_dir, name))