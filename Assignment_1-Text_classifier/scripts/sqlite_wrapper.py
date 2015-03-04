import sqlite3
from nltk import word_tokenize
from collections import defaultdict


class SqliteWrapper:
    def __init__(self, dbname='../assignment.db'):
        self.executor = None
        self._init_db_connection(dbname)

    def finish(self):
        self.connection.commit()
        self.executor.close()

    def executemany(self, sql, args):
        self.executor.executemany(sql, args)

    def execute(self, sql, args=''):
        self.executor.execute(sql, args)

    def _init_db_connection(self, dbname):
        if self.executor is None:
            self.connection = sqlite3.connect(dbname)
            self.executor = self.connection.cursor()


class SqliteFiller:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def fill_tables(self, data_path, labels_path):
        self._fill_sents(data_path, labels_path)
        self._fill_word_freqs()
        self._make_test_set()
        self._make_stopword_list()
        self._delete_test_set_from_sents()

    def make_tables(self):
        self._execute('''CREATE TABLE words_freq (label TEXT, word TEXT, count INT)''')
        self._execute('''CREATE TABLE sents (label TEXT, sent TEXT)''')
        self._execute('''CREATE TABLE test_sents (id INTEGER PRIMARY KEY, label TEXT, sent TEXT)''')
        self._execute('''CREATE TABLE stop_words (word TEXT)''')

    def _delete_test_set_from_sents(self):
        self._execute("SELECT * FROM test_sents")
        for row in list(self.executor):
            label = row[1]
            sent = row[2]
            self._execute("DELETE FROM sents where sent=? and label=?", (sent, label))

    def _make_test_set(self):
        labels = ['Rest', 'Business', 'PrivateLife', 'Entertainment']
        limit = 50000
        data = {}
        for label in labels:
            self._execute("SELECT * FROM sents WHERE label=? ORDER BY RANDOM() LIMIT ?",
                         (label, limit))
            data[label] = list(self.executor)

        id = 1
        insert_data = []
        for label, rows in data.items():
            for row in rows:
                insert_data.append((id, label, row[1]))
                id += 1
                print(id)
        self._executemany("INSERT INTO test_sents VALUES(?, ?, ?)", insert_data)

    def _make_stopword_list(self):
        labels = ['Rest', 'Business', 'PrivateLife', 'Entertainment']
        limit = 100
        data = {}
        for label in labels:
            self._execute("SELECT * FROM words_freq WHERE label=? ORDER BY count desc LIMIT ?",
                         (label, limit))
            data[label] = {}
            data[label]['all'] = list(self.executor)
            data[label]['words'] = set(row[1] for row in data[label]['all'])

        union_set = set()
        for label, _ in data.items():
            union_set = union_set.union(data[label]['words'])

        intersection_set = union_set.copy()
        for label, _ in data.items():
            intersection_set = intersection_set.intersection(data[label]['words'])

        words = [(word, ) for word in intersection_set]
        self._executemany("INSERT INTO stop_words VALUES(?)", words)

    def _execute(self, sql, args):
        self.wrapper.execute(sql, args)

    def _executemany(self, sql, args):
        self.wrapper.executemany(sql, args)

    def _fill_word_freqs(self):
        words_freq = self._get_words_freq()
        self._executemany("INSERT INTO words_freq VALUES(?, ?, ?)", words_freq)

    def _get_words_freq(self):
        self._execute("SELECT * FROM sents")
        freqs = defaultdict(int)
        for ind, row in enumerate(self.executor):
            label = row[0]
            sent = row[1]
            for word in sent.split():
                freqs[(label, word)] += 1
            print(ind)
        return [(label_sent[0], label_sent[1], count) for label_sent, count in freqs.items()]

    def _fill_sents(self, data_path, labels_path):
        labels_with_sents = self._get_labels_with_sents(data_path, labels_path)
        self.executemany("INSERT INTO sents VALUES(?, ?)", labels_with_sents)

    def _get_labels_with_sents(self, data_path, labels_path):
        sents = None
        with open(data_path) as f:
            sents = [sent.rstrip() for sent in f.readlines()]

        labels = None
        with open(labels_path) as f:
            labels = [label.rstrip() for label in f.readlines()]

        return list(zip(labels, sents))




if __name__ == '__main__':
    importer = SqliteWrapper()

    importer.finish()
