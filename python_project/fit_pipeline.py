import argparse
import os
import joblib
import logging
import logging.config

from python_project.loader import DataLoader
from python_project.feature import FeatureEngineering
from python_project.model import Model


def arg_parser():
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Simple project example')
    parser.add_argument('--fe-path',
                        type=str,
                        help="Where to save the feature engineering pipeline",
                        dest='fe_pipeline_save_path',
                        required=False,
                        default=os.path.join(current_working_dir,
                                             "../models/fe_pipeline/fe_pipeline.joblib"))
    parser.add_argument('--classifier-path',
                        type=str,
                        help="Where to save the resulting classifier",
                        dest='classifier_save_path',
                        required=False,
                        default=os.path.join(current_working_dir,
                                             "../models/classifier/classifier.joblib"))
    parser.add_argument('--use-idf',
                        type=bool,
                        help='Use IDF in the TF-IDF',
                        dest='use_idf',
                        required=True)

    return parser


def main():
    logging.config.fileConfig('%s/../logging.conf' % os.path.dirname(os.path.abspath(__file__)))
    logger = logging.getLogger(name="simpleExample")

    parser = arg_parser()
    args = parser.parse_args()

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    fe_pipeline_save_path = args.fe_pipeline_save_path
    classifier_save_path = args.classifier_save_path

    logger.info("Create Directories to save models")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, '../models/'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, '../models/fe_pipeline/'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, '../models/classifier/'), exist_ok=True)

    logger.info("Loading NewsGroup Data")
    newsgroup_data_loader = DataLoader(categories=categories)
    twenty_train, _ = newsgroup_data_loader.load_data()

    logger.info("Fit and Save Feature Engineering Pipeline")
    fe = FeatureEngineering(args.use_idf)
    fe.fit_and_save_pipeline(
        df_train=twenty_train.data,
        save_path=fe_pipeline_save_path)

    logger.info("Load Feature Engineering Pipeline and apply transformations on train set")
    fe_pipeline = joblib.load(fe_pipeline_save_path)
    X_train_tfidf = fe_pipeline.transform(twenty_train.data)

    logger.info("Training a classifier")
    model = Model()
    model.fit_and_save_model(
        df_train=X_train_tfidf,
        labels_train=twenty_train.target,
        save_path=classifier_save_path
    )

    logger.info("Model trained and saved")


if __name__ == "__main__":
    main()
