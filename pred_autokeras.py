from autokeras.utils import pickle_from_file
from autokeras import ImageClassifier
from load_utils import *
import numpy as np

def dog_submit():
	X_train, y_train, X_test = load_dog_breed()

	target = pd.read_csv("datasets/dog-breed/sample_submission.csv")

	column = target.columns.values[1:]

	id = target.id

	model = pickle_from_file("best_auto_keras_model_dog.h5")
	results = model.predict_proba(X_test)

	df = pd.DataFrame(columns=column,data=results)

	df.insert(0,"id",id,True)

	print(df.head())

	df.to_csv("dog_auto_keras_submission.csv", index=False)


def invasive_submit():
        X_train, y_train, X_test = load_invasive_species()

        target = pd.read_csv("datasets/invasive-species/sample_submission.csv")

        column = target.columns.values[1:]

        id = np.array(dtype=int,object=target.name)

        print(id.shape)
        print(X_test.shape)

        model = pickle_from_file("best_auto_keras_model_invasive.h5")
        results = model.predict(X_test)

        df = pd.DataFrame(columns=column,data=results)
	
        df.insert(0,"name",id,True)

        print(df.head())

        df.to_csv("invasive_auto_keras_submission.csv", index=False)


def plant_seedlings():
        X_train, y_train, X_test = load_plant_seedlings()

        target = pd.read_csv("datasets/plant-seedlings/sample_submission.csv")

        column = target.columns.values[1:]

        id = np.array(target.file)
        print(id.shape)
        print(X_test.shape)

        model = pickle_from_file("best_auto_keras_model_plant.h5")
        results = model.predict(X_test)

        df = pd.DataFrame(columns=column,data=results)

        df.insert(0,"id",id,True)

        print(df.head())

        df.to_csv("plant_auto_keras_submission.csv", index=False)


#plant_seedlings()
#invasive_submit()
dog_submit()
