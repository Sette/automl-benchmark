import logging
import threading
import time

from tpot import TPOTClassifier
from fi_utils import *
from benchmark_utils import timer
import autosklearn.classification
import pandas as pd
from hpsklearn import HyperoptEstimator


all_datasets = [
        ("dont_overfit", load_dont_overfit),
        ("porto_seguro", load_porto_seguro),
        ("santander_customer", hyperopt_fit_pred),
        ("microsoft_malware", load_microsoft_malware),
    ]

def tpot_fit_pred(X_train,y_train,X_test):
    
    start_time = timer(None)
    tp.fit(X_train, y_train)
    tp.export('tpot_pipeline_dont_overfit.py')
    preds = tp.predict(X_test)
    submission_time.append((name,timer(start_time)))

    submission = pd.DataFrame({
        "id": X_test["id"],
        "target": preds
    })

    submissions.append(("tpot",submission))


def autosk_fit_pred(X_train,y_train,X_test):
    start_time = timer(None)
    ak.fit(X_train.copy(), y_train.copy(), dataset_name='dont_overfit')
    ak.refit(X_train.copy(), y_train.copy())
    preds =  ak.predict(X_test)

    submission_time.append((name,timer(start_time)))

    submission = pd.DataFrame({
        "id": X_test["id"],
        "target": preds
    })

    submissions.append(("autosk",submission))

def hyperopt_fit_pred(X_train,y_train,X_test):
    start_time = timer(None)
    hp.fit(X_train.as_matrix(),y_train.as_matrix())
    preds =  hp.predict(X_test)

    submission_time.append((name,timer(start_time)))

    submission = pd.DataFrame({
        "id": X_test["id"],
        "target": preds
    })

    submissions.append(("hyperopt",submission))
    

def thread_function(name):
    logging.info("Thread %s: starting", name)
    print("Realiza treinamento para com uma ferramenta específica")
    logging.info("Thread %s: finishing", name)

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")


threads = list()

all_models = [
        ("autosk", autosk_fit_pred),
        ("tpot", tpot_fit_pred),
        ("hyperopt", hyperopt_fit_pred)
    ]

for name_dataset, dataset in all_datasets:

    tp = TPOTClassifier(verbosity=2)
    ak = autosklearn.classification.AutoSklearnClassifier()
    hp = HyperoptEstimator()
    submissions = []
    submission_time = []

    X_train, X_test, y_train = dataset()

    for name, model in all_models:
        print("Training with ", name)
        start_time = timer(None)
        x = threading.Thread(target=model, args=(X_train.copy(),y_train.copy(),X_test.copy()))
        threads.append(x)
        x.start()
        
    for name, sub in submissions:
        sub.to_csv(name_dataset+'_'+name+'_submission.csv', index=False)

        
    for name, sub in submission_time:
        sub.to_csv(name_dataset+'_'+name+'_submission_time.csv', index=False)
