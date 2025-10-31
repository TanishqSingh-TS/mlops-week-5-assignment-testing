Project_Id = "keen-phalanx-473718-p1"
Location = "us-central1"  
Bucket_URI = "gs://mlops-course-keen-phalanx-473718-p1"

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import joblib
from zoneinfo import ZoneInfo
from sklearn import metrics
import mlflow
import requests
import sys


mlflow.set_tracking_uri("http://34.132.101.61:5000/")
mlflow.sklearn.autolog(
    max_tuning_runs=15,
    registered_model_name="iris-tree-classifier"
)


data = pd.read_csv("./data/iris.csv")
train, test = train_test_split(data, test_size = 0.2, stratify = data['species'], random_state = 55)

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight':[None,'balanced']
}


with mlflow.start_run(run_name="DecisionTree Classifier Hyperparameter Tuning"):
    model = DecisionTreeClassifier(random_state = 1)
    grid_search = GridSearchCV(model,param_grid,cv=5,scoring="accuracy",n_jobs=-1,verbose=1)
    grid_search.fit(X_train,y_train)
    best_score = grid_search.score(X_test,y_test)
    print(f"Best Parameters for Model: {grid_search.best_params_}")
    print(f"Best Cross Validation Score for Model: {grid_search.best_score_:.4f}")
    print(f"Test Score for Model: {best_score:.3f}")
    




    
