from tpot import TPOTClassifier
from fi_utils import load_dont_overfit
from benchmark_utils import timer
import autosklearn.classification
import pandas as pd
from hpsklearn import HyperoptEstimator



X_train, X_test, y_train = load_dont_overfit()

tp = TPOTClassifier(verbosity=2)
ak = autosklearn.classification.AutoSklearnClassifier()
hp = HyperoptEstimator()

def tpot_fit_pred(X_train,y_train,X_test):
    tp.fit(X_train, y_train)
    tp.export('tpot_pipeline_dont_overfit.py')
    return tp.predict(X_test)

def autosk_fit_pred(X_train,y_train,X_test):
    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    ak.fit(X_train.copy(), y_train.copy(), dataset_name='dont_overfit')
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    ak.refit(X_train.copy(), y_train.copy())
    return ak.predict(X_test)

def hyperopt_fit_pred(X_train,y_train,X_test):
    hp.fit(X_train,y_train)
    
    return hp.predict(X_test)
    

all_models = [
        ("hyperopt", hyperopt_fit_pred),
        ("autosk", autosk_fit_pred),
        ("tpot", tpot_fit_pred),
]


submission_time = []
for name, model in all_models:
    print("Training with ", name)
    start_time = timer(None)
    preds = model(X_train.copy(),y_train.copy(),X_test.copy())
    submission_time.append((name,timer(start_time)))
    submission = pd.DataFrame({
        "id": X_test["id"],
        "target": preds
    })
    
    submission.to_csv('submission_'+name+'.csv', index=False)

sub_time = pd.DataFrame({
    "name": submission_time[0],
    "time": submission_time[1]

})

sub_time.to_csv('submission_time.csv', index=False)
