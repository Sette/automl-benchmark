from load_utils import *
import h2o
from h2o.automl import H2OAutoML


h2o.init()

all_datasets = [
        ("house_prices", load_house_prices),
        #("santander_value", load_santander_value),
        #("taxi_fare", load_taxi_fare),
]


for name_dataset, dataset in all_datasets:
	X_train, y_train, X_test, id_test, id_name,target_name = dataset()
	print(y_train.values)
	X_train_cp = X_train.copy()
	X_train_cp[target_name] = y_train
	train = h2o.H2OFrame.from_python(X_train_cp)
	test = h2o.H2OFrame.from_python(X_test)
	# Identify predictors and response
	x = train.columns
	y = target_name
    	# (limited to 1 hour max runtime by default)
	aml = H2OAutoML()
	aml.train(x=x, y=y, training_frame=train)
	preds = aml.predict(test).as_data_frame().values
	print(preds)
