from numpy import loadtxt
from keras.models import load_model

from autokeras import ImageClassifier
from load_utils import *
import numpy as np

def dog_submit():
        X_train, y_train, X_test = load_dog_breed()

        target = pd.read_csv("datasets/dog-breed/sample_submission.csv")

        column = target.columns.values[1:]

        id = target.id
        
        model = load_model("best_auto_keras_model_dog.h5")        
        model.summary()
        results = model.predict_proba(X_test)
        
        df = pd.DataFrame(columns=column,data=results)

        df.insert(0,"id",id,True)

        print(df.head())

        df.to_csv("dog_auto_keras_submission.csv", index=False)

dog_submit()
