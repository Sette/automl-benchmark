
from load_utils import *
import pandas as pd
from hpsklearn import HyperoptEstimator

hp = HyperoptEstimator()
X_train, y_train, X_test = load_dont_overfit()

hp.fit(X_train.as_matrix(),y_train.as_matrix())
preds =  hp.predict(X_test.as_matrix())

print(preds)



