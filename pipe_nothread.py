import logging
import threading
import time

from tpot import TPOTClassifier
from load_utils import *
from benchmark_utils import timer
import autosklearn.classification
import pandas as pd
from hpsklearn import HyperoptEstimator

all_datasets = [
        ("dont_overfit", load_dont_overfit),
        ("porto_seguro", load_porto_seguro),
        ("santander_customer", load_santander_customer),
        ("microsoft_malware", load_microsoft_malware)
    ]

submissions = []
threads = list()

def tpot_fit_pred(X_train,y_train,X_test,id_test,name,name_dataset):    
    start_time = timer(None)
    tp.fit(X_train, y_train)
    tp.export('tpot_pipeline_dont_overfit.py')
    preds = tp.predict(X_test)


    submission_time = pd.DataFrame({
        "name": name,
        "name_dataset": name_dataset,
        "time":timer(start_time)
    })

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission_time.sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)

    submission.sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)
    

    submissions.append(("tpot",submission))


def autosk_fit_pred(X_train,y_train,X_test,id_test,name,name_dataset):
    start_time = timer(None)
    ak.fit(X_train, y_train)
    ak.refit(X_train, y_train)
    preds =  ak.predict(X_test)

    submission_time = pd.DataFrame({
        "name": name,
        "name_dataset": name_dataset,
        "time":timer(start_time)
    })

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission_time.sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)

    submission.sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)

def hyperopt_fit_pred(X_train,y_train,X_test,id_test,name,name_dataset):
    start_time = timer(None)
    hp.fit(X_train.as_matrix(),y_train.as_matrix())
    time = timer(start_time)
    print(time)
    preds =  hp.predict(X_test.as_matrix())
    
    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission.sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)
    

all_models = [
    ("hyperopt", hyperopt_fit_pred),
    ("autosk", autosk_fit_pred),
    ("tpot", tpot_fit_pred)
]

for name_dataset, dataset in all_datasets:

    tp = TPOTClassifier(verbosity=2)
    ak = autosklearn.classification.AutoSklearnClassifier()
    hp = HyperoptEstimator()
    submissions = []
    submission_time = []

    X_train, y_train, X_test, id_test = dataset()


    for name, model in all_models:
        print("Training with ", name)
        model(X_train,y_train,X_test,id_test,name,name_dataset)
        