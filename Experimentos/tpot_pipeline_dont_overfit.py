import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.8039999999999999
exported_pipeline = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.2, n_estimators=100), step=0.7000000000000001),
    StandardScaler(),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.8, n_estimators=100), step=0.4),
    LogisticRegression(C=1.0, dual=False, penalty="l1")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
