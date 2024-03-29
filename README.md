# Python project template

*Boilerplate template for machine learning projects in Python.*

To build docker image :
```sh
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f Dockerfile .
```

To run model fit :
```sh
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=python_project/fit_pipeline.py
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python ${PYTHON_SCRIPT} ${PARAMS}
```
This script will take parameters :
```
usage: fit_pipeline.py [-h] [--fe-path FE_PIPELINE_SAVE_PATH]
                        [--classifier-path CLASSIFIER_SAVE_PATH] --use-idf
                        USE_IDF
 
Simple project example
 
optional arguments:
  -h, --help            show this help message and exit
  --fe-path FE_PIPELINE_SAVE_PATH
                        Where to save the feature engineering pipeline
  --classifier-path CLASSIFIER_SAVE_PATH
                        Where to save the resulting classifier
  --use-idf USE_IDF     Use IDF in the TF-IDF
```
 
To run prediction
```sh
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=python_project/predict.py
```
This script will take parameters :
```
usage: predict.py [-h] [--fe-path FE_PIPELINE_SAVE_PATH]
                       [--classifier-path CLASSIFIER_SAVE_PATH]
  
Simple project example
  
optional arguments:
  -h, --help            show this help message and exit
  --fe-path FE_PIPELINE_SAVE_PATH
                        Where to save the feature engineering pipeline
  --classifier-path CLASSIFIER_SAVE_PATH
                        Where to save the resulting classifier
```

To run tests
```sh
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python setup.py test
```
