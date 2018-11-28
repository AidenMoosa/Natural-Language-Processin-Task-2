import time
from os import chdir, listdir, rename
from os.path import dirname, join

working_dir = dirname(__file__)
chdir(working_dir)

from nltk import download, word_tokenize
download("punkt")

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

def preprocess_reviews_to_files(review_dir, pickle_dir):
    reviews = listdir(review_dir)
    for i, review in enumerate(reviews):
        if i < 12500:
            continue
        text = open(join(review_dir, review), 'r', encoding="utf-8").read()
        token_list = word_tokenize(text)
        doc = TaggedDocument(token_list, [review])
        pickle.dump(doc, open(join(pickle_dir, review) + ".p", "wb"))

def load_documents(pickle_dir):
    documents = listdir(pickle_dir)
    docs = []
    for document in documents:
        doc = pickle.load(open(join(pickle_dir, document), "rb"))
        docs.append(doc)
    return docs

def get_stratified_split(pos_dir, neg_dir, num_folds, offset):
    pos_reviews = [join(pos_dir, f) for f in listdir(pos_dir)]
    neg_reviews = [join(neg_dir, f) for f in listdir(neg_dir)]
    train_reviews = []
    test_reviews = []

    i = 0
    while i < len(pos_reviews):
        test_reviews.append(pos_reviews[i + offset])
        for j in range(1, num_folds):
            index = i + ((j + offset) % num_folds)
            train_reviews.append(pos_reviews[index])
        i += num_folds

    i = 0
    while i < len(neg_reviews):
        test_reviews.append(neg_reviews[i + offset])
        for j in range(1, num_folds):
            index = i + ((j + offset) % num_folds)
            train_reviews.append(neg_reviews[i + (j % num_folds)])
        i += num_folds

    return (train_reviews, test_reviews)

def main():
    start_time = time.process_time()

    #preprocess_reviews_to_files("data/neg", "data/neg_tokenized")
    #preprocess_reviews_to_files("data/pos", "data/pos_tokenized")
    #preprocess_reviews_to_files("data/unsup", "data/unsup_tokenized")

    neg_docs = load_documents("data/neg_tokenized")
    pos_docs = load_documents("data/pos_tokenized")
    unsup_docs = load_documents("data/unsup_tokenized")
    all_docs = neg_docs + pos_docs + unsup_docs

    model = Doc2Vec(all_docs, vector_size=5, window=2, min_count=1, workers=4)
    vector = model.infer_vector(["system", "response"])

    print("running time is " + str(time.process_time() - start_time))

def rename_files(file_dir, output_dir):
    files = listdir(file_dir)
    for i, f in enumerate(files):
        name = str(12500 + i) + "_" + f.split("_")[1]
        rename(join(file_dir, f), join(output_dir, name))

main()

'''
from nltk.stem import PorterStemmer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

ps = PorterStemmer()

###### THINGS TO CHANGE ######
presence = True
stem = True
num_folds = 10
unigrams = False
bigrams = True
feature_cutoff = 7
##############################

for i in range(num_folds):
    print("Splitting reviews into training/test sets: " + str(i+1) + "/" + str(num_folds))
    (train_reviews, test_reviews) = get_stratified_split(pos_dir, neg_dir, num_folds, i)

    print("Extracting training features...")
    train_length = len(train_reviews)
    train_labels = np.zeros(train_length)
    train_labels[train_length//2:train_length] = 1
    train_matrix = extract_features_presence(train_reviews) if presence else extract_features(train_reviews)

    print("Training SVM model...")
    model2 = SVC(kernel="linear")
    model2.fit(train_matrix, train_labels)

    print("Extracting test features...")
    test_length = len(test_reviews)
    test_labels = np.zeros(test_length)
    test_labels[test_length//2:test_length] = 1
    test_matrix = extract_features_presence(test_reviews) if presence else extract_features(test_reviews)

    result2 = model2.predict(test_matrix)
    scores2.append(accuracy_score(test_labels, result2))

print("")
print("Features:" + " Unigrams" if unigrams else "" + " Bigrams" if bigrams else "")
print("No. Features: " + str(len(dictionary)))
print("Frequency or Presence?: " + "Presence" if presence else "Frequency")
print("Stemmed?: " + "Yes" if stem else "No")
print("NB Score: " + str(np.mean(scores1)))
print("SVM Score: " + str(np.mean(scores2)))
print("P-Value: " + str(p))
'''