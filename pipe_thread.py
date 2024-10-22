import threading
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
from load_utils import *
from benchmark_utils import timer
import autosklearn.classification
from hpsklearn import HyperoptEstimator
import pandas as pd

h2o.init()

all_datasets = [
        #("microsoft_malware", load_microsoft_malware),
        ("porto_seguro", load_porto_seguro),
        ("santander_customer", load_santander_customer)
    ]

submissions = []
threads = list()

def h20_fit_pred(X_train,y_train,X_test,id_test,name_dataset):
    X_train_cp = X_train.copy()
    X_train_cp['target'] = y_train
    start_time = timer(None)
    train = h2o.H2OFrame.from_python(X_train)
    test = h2o.H2OFrame.from_python(X_test)
    
    # Identify predictors and response
    x = train.columns
    y = "target"

    # (limited to 1 hour max runtime by default)
    aml = H2OAutoML()
    aml.train(x=x, y=y, training_frame=train)
    time = timer(start_time)
    preds = aml.predict(test).as_data_frame()
    preds_final = [1 if x> 0.5 else 0 for x in preds.values]

    time_out = open(name_dataset+'_'+'h2o',"w")     
    time_out.write(time) 
    time_out.close() 

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds_final
    })

    submission.to_csv(name_dataset+'_'+'h2o'+'_submission.csv', index=False)


def tpot_fit_pred(X_train,y_train,X_test,id_test,name_dataset):    
    tp = TPOTClassifier(verbosity=3)
    start_time = timer(None)
    tp.fit(X_train, y_train)
    tp.export('tpot_pipeline_dont_overfit.py')
    time = timer(start_time)
    preds = tp.predict(X_test)

    time_out = open(name_dataset+'_'+'tpot',"w") 
    time_out.write(time) 
    time_out.close() 

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission.to_csv(name_dataset+'_'+'tpot'+'_submission.csv', index=False)


def autosk_fit_pred(X_train,y_train,X_test,id_test,name_dataset):
    ak = autosklearn.classification.AutoSklearnClassifier(ml_memory_limit=51000)
    start_time = timer(None)
    ak.fit(X_train.copy(), y_train.copy())
    ak.refit(X_train.copy(), y_train.copy())
    time = timer(start_time)
    preds =  ak.predict(X_test.copy())

    time_out = open(name_dataset+'_'+'autosk',"w") 
    time_out.write(time) 
    time_out.close() 

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission.to_csv(name_dataset+'_'+'autosk'+'_submission.csv', index=False)
    

def hyperopt_fit_pred(X_train,y_train,X_test,id_test,name_dataset):
    hp = HyperoptEstimator()
    start_time = timer(None)
    hp.fit(X_train.values,y_train.values)
    time = timer(start_time)
    preds =  hp.predict(X_test.values)
    
    time_out = open(name_dataset+'_'+'hyperopt',"w") 
    time_out.write(time) 
    time_out.close() 

    submission = pd.DataFrame({
        "id": id_test,
        "target": preds
    })

    submission.to_csv(name_dataset+'_'+'hyperopt'+'_submission.csv', index=False)
    

all_models = [
    ("tpot", tpot_fit_pred),
    #("autosk", autosk_fit_pred),
    #("hyperopt", hyperopt_fit_pred),
    #('h2o',h20_fit_pred),
]

for name_dataset, dataset in all_datasets:

    submissions = []
    submission_time = []

    X_train, y_train, X_test, id_test = dataset()

    for name, model in all_models:
        print("Training with ", name, ' in dataset: ', name_dataset)
        try:
            x = threading.Thread(target=model, args=(X_train,y_train,X_test,id_test,name_dataset))
            threads.append(x)
            x.start()
        except Exception as e:
            error_out = open('error_'+name_dataset+'_'+name,"w") 
            print(e) 
            error_out.write(str(e))
            error_out.close() 
            print("Erro no expermento. dataset: ", name_dataset, "automl: ", name)
        

    for thread in threads:
        print("Aguardando threads")
        thread.join()

    
    print("Fim das threads")
    