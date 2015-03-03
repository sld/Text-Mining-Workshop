import sqlite3
from nltk import word_tokenize
from collections import defaultdict


class SqliteWrapper:
    def __init__(self, dbname='../assignment.db'):
        self.executor = None
        self._init_db_connection(dbname)

    def make_tables(self):
        self.executor.execute('''CREATE TABLE words_freq (label TEXT, word TEXT, count INT)''')
        self.executor.execute('''CREATE TABLE sents (label TEXT, sent TEXT)''')

    def fill_tables(self, data_path, labels_path):
        # self._fill_sents()
        self._fill_word_freqs()

    def finish(self):
        self.connection.commit()
        self.executor.close()

    def executemany(self, sql, args):
        self.executor.executemany(sql, args)

    def execute(self, sql, args=''):
        self.executor.execute(sql, args)

    def _fill_word_freqs(self):
        words_freq = self._get_words_freq()
        self.executemany("INSERT INTO words_freq VALUES(?, ?, ?)", words_freq)

    def _get_words_freq(self):
        self.execute("SELECT * FROM sents")
        freqs = defaultdict(int)
        for ind, row in enumerate(self.executor):
            label = row[0]
            sent = row[1]
            for word in sent.split():
                freqs[(label, word)] += 1
            print(ind)
        return [(label_sent[0], label_sent[1], count) for label_sent, count in freqs.items()]

    def _fill_sents(self):
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

    def _init_db_connection(self, dbname):
        if self.executor is None:
            self.connection = sqlite3.connect(dbname)
            self.executor = self.connection.cursor()


if __name__ == '__main__':
    importer = SqliteWrapper()
    # importer.make_tables()
    # importer.fill_tables('../texts_train_10_full.txt', '../labels_train_10_full.txt')
    importer.finish()
