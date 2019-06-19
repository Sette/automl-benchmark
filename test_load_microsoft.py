
from fi_utils import *
import pandas as pd
from hpsklearn import HyperoptEstimator

hp = HyperoptEstimator()
X_train, X_test, y_train = load_microsoft_malware()
print(X_train.head())
