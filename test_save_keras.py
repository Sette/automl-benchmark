from autokeras.utils import pickle_from_file
from autokeras import ImageClassifier
from load_utils import *
import numpy as np

def dog_save():
        model = pickle_from_file("best_auto_keras_model_dog.h5")
        model_v = model.graph.produce_model()
        model_v.eval()
        model.save("modelo_dog")

dog_save()
