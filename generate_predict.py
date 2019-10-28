from autokeras.utils import pickle_from_file
from load_utils import *
import pandas as pd

model = pickle_from_file('best_auto_keras_model_plant.h5')
x_train,y_train,x_test = load_invasive_species()
results = model.predict(x_test)
print(results)


submission = pd.DataFrame({
        "target": results
})

submission.to_csv('invasive_'+'autokeras_submission.csv', index=False)
