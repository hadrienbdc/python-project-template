from sklearn.naive_bayes import MultinomialNB
import joblib


class Model:
    def __init__(self):
        self.clf = MultinomialNB()

    @staticmethod
    def fit_model(clf, df_train, labels_train):
        clf.fit(df_train, labels_train)

    @staticmethod
    def save_model(clf, save_path):
        joblib.dump(clf, save_path)

    def fit_and_save_model(self, df_train, labels_train, save_path):
        self.fit_model(self.clf, df_train, labels_train)
        self.save_model(self.clf, save_path)
