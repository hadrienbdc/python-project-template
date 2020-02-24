from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import joblib


class FeatureEngineering:
    def __init__(self, use_idf):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer(use_idf=use_idf))
        ])

    @staticmethod
    def fit_pipeline(pipeline, df_train):
        pipeline.fit(df_train)

    @staticmethod
    def save_pipeline(pipeline, save_path):
        joblib.dump(pipeline, save_path)

    def fit_and_save_pipeline(self, df_train, save_path):
        self.fit_pipeline(self.pipeline, df_train)
        self.save_pipeline(self.pipeline, save_path)
