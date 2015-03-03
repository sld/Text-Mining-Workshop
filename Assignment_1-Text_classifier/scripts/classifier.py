import nltk.corpus
import pickle

from sys import argv
from collections import defaultdict
from sqlite_wrapper import SqliteWrapper
from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from nltk.probability import FreqDist
from nltk.classify import accuracy

from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class Classifier:

    def __init__(self):
        self.sqlite_wrapper = SqliteWrapper()
        self._labels = ['Rest', 'Business', 'PrivateLife', 'Entertainment']

    def pipeline(self):
        train, test = self._load_from_file('small')
        self.train_classifier(train)
        print(self.test_classifier(test))

    def train_classifier(self, data):
        sents, labels = self._split(data)
        pipeline_ = Pipeline([('counts', CountVectorizer(ngram_range=(1, 3))),
                              ('tfidf', TfidfTransformer()),
                              ('classifier', LinearSVC())])
        pipeline_.fit(sents, labels)
        self.classifier_pipeline = pipeline_

    def test_classifier(self, data):
        sents, labels = self._split(data)
        predicted = self.classifier_pipeline.predict(sents)
        return accuracy_score(labels, predicted)

    def _load_from_file(self, size):
        return pickle.load(open('../train_and_test_{0}.p'.format(size), 'rb'))

    def _split(self, data):
        sents = [e[0] for e in data]
        labels = [e[1] for e in data]
        return sents, labels

    def _train_and_test_sets(self, limit=300000):
        data = {}
        for label in self._labels:
            self.sqlite_wrapper.execute("SELECT * FROM sents WHERE label=? ORDER BY RANDOM() LIMIT ?",
                                        (label, limit))
            data[label] = list(self.sqlite_wrapper.executor)

        train_data = []
        test_data = []
        cnt80 = int(0.8*limit)
        for label, sents in data.items():
            classifier_data = [(self._features(sent), label) for _, sent in sents]
            train_data += classifier_data[:cnt80]
            test_data += classifier_data[cnt80:]
        # pickle.dump((train_data, test_data), open('../train_and_test.p', 'wb'))
        return (train_data, test_data)

    def _features(self, sent):
        return sent


if __name__ == '__main__':
    cl = Classifier()
    cl.pipeline()
