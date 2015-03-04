import pickle

from sys import argv
from collections import defaultdict

from sqlite_wrapper import SqliteWrapper
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class BaseClassifier:
    def __init__(self):
        self.sqlite_wrapper = SqliteWrapper()
        self._labels = ['Rest', 'Business', 'PrivateLife', 'Entertainment']
        self._test_data_filled = False

    def _split(self, data):
        sents = [e[0] for e in data]
        labels = [e[1] for e in data]
        return sents, labels

    def _features(self, sent):
        return sent

    def _get_batch(self, i, limit=1000, query=None, col_num=1):
        if query is None:
            query = "SELECT * FROM sents WHERE label=? ORDER BY sent LIMIT ? OFFSET ?"
        data = {}
        offset = limit*i
        for label in self._labels:
            self.sqlite_wrapper.execute(query, (label, limit, offset))
            data[label] = list(self.sqlite_wrapper.executor)

        classifier_data = []
        for label, sents in data.items():
            classifier_data += [(self._features(row[col_num]), label) for row in sents]
        return classifier_data

    def _get_stop_words(self):
        self.sqlite_wrapper.execute("select * from stop_words")
        return [e[0] for e in list(self.sqlite_wrapper.executor)]


class StreamClassifier(BaseClassifier):
    def pipeline(self, max_i=400, batch_size=5000, test_iteration=20):
        classifier_prerformance = []
        self._init_classifier()
        self._online_train_and_test(max_i, batch_size, test_iteration)

    def _init_classifier(self):
        self.vectorizer = HashingVectorizer(ngram_range=(1, 3),
                                            non_negative=True,
                                            stop_words=self._get_stop_words())
        self.classifier = MultinomialNB()

    def _online_train_and_test(self, max_i, batch_size, test_iteration):
        classifier_prerformance = []
        for i in range(0, max_i):
            batch = self._get_batch(i, batch_size)
            sents, labels = self._split(batch)
            transformed_sents = self.vectorizer.transform(sents)

            self.classifier.partial_fit(transformed_sents, labels, classes=self._labels)

            if i % test_iteration == 0:
                scores = self._test_classifier(i)
                pickle.dump((self.classifier, self.vectorizer, i, max_i),
                            open('../states/stream_classifier_{0}.p'.format(i), 'wb'))

                classifier_prerformance.append((i, batch_size, scores))
                pickle.dump(classifier_prerformance,
                            open('../states/overall_stream_classifier_performance.p', 'wb'))
                print((i, batch_size, scores))

    def _test_classifier(self, i):
        if self._test_data_filled is not True:
            query = "SELECT * FROM test_sents WHERE label=? ORDER BY sent LIMIT ? OFFSET ?"
            test_data = self._get_batch(0, 10000, query, 2)
            test_sents, self.test_labels = self._split(test_data)
            self.test_transformed_sents = self.vectorizer.transform(test_sents)
            self.test_data_filled = True

        predicted = self.classifier.predict(self.test_transformed_sents)
        return accuracy_score(self.test_labels, predicted), f1_score(self.test_labels, predicted)


class InMemoryClassifier(BaseClassifier):
    def pipeline(self, size=100000):
        train = self._get_batch(0, size)
        test = self._get_batch(1, size)
        self.train_classifier(train)
        print(self.test_classifier(test))

    def train_classifier(self, data):
        sents, labels = self._split(data)
        pipeline_ = Pipeline([('counts', HashingVectorizer(ngram_range=(1, 3),
                                                           non_negative=True,
                                                           stop_words=self._get_stop_words())),
                              ('tfidf', TfidfTransformer()),
                              ('classifier',  LinearSVC())])
        pipeline_.fit(sents, labels)
        self.classifier_pipeline = pipeline_

    def test_classifier(self, data):
        sents, labels = self._split(data)
        predicted = self.classifier_pipeline.predict(sents)
        return accuracy_score(labels, predicted), f1_score(labels, predicted)

    def _load_from_file(self, size):
        return pickle.load(open('../train_and_test_{0}.p'.format(size), 'rb'))

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
        return (train_data, test_data)


if __name__ == '__main__':
    # cl_in_memory = InMemoryClassifier()
    # cl_in_memory.pipeline(100000)

    cl_stream = StreamClassifier()
    cl_stream.pipeline(1035, 2500)
