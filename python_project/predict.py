import argparse
import os
import joblib
import numpy as np
import logging
import logging.config

from python_project.loader import DataLoader


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

    return parser


def main():
    logging.config.fileConfig('%s/../logging.conf' % os.path.dirname(os.path.abspath(__file__)))
    logger = logging.getLogger(name="simpleExample")

    parser = arg_parser()
    args = parser.parse_args()

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    fe_pipeline_save_path = args.fe_pipeline_save_path
    classifier_save_path = args.classifier_save_path

    logger.info("Loading NewsGroup Data")
    newsgroup_data_loader = DataLoader(categories=categories)
    _, twenty_test = newsgroup_data_loader.load_data()

    logger.info("Load Feature Engineering Pipeline and apply transformations on train set")
    fe_pipeline = joblib.load(fe_pipeline_save_path)
    X_test_tfidf = fe_pipeline.transform(twenty_test.data)

    logger.info("Load classifier and apply it")
    clf = joblib.load(classifier_save_path)
    predicted = clf.predict(X_test_tfidf)

    logger.info("Accuracy : {}".format(np.mean(predicted == twenty_test.target)))


if __name__ == "__main__":
    main()
