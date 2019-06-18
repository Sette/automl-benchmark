
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
from fi_utils import load_dont_overfit
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

X_train, X_test, y_train = load_dont_overfit()

automl = autosklearn.classification.AutoSklearnClassifier()

# fit() changes the data in place, but refit needs the original data. We
 # therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy(), dataset_name='dont_overfit')
# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(X_train.copy(), y_train.copy())

print(automl.show_models())

predictions = automl.predict(X_test)

submission_autosk = pd.DataFrame({
        "id": X_test["id"],
        "target": predictions
    })

submission_autosk.to_csv('submission_autosk.csv', index=False)

#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))