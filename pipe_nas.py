# To run first unzip ../load_raw_image_data.zip into ../
# so the train and test directories reside in this directory 

from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset

import adanet
import tensorflow as tf

data_dir = "datasets/dog-breed"

def load_images():
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/labels_real.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission_real.csv",
                                        images_path=data_dir+"/test")
    print(x_test[0].shape)
    
    return x_train, y_train,x_test[0]
    


def run_autokeras():
    x_train, y_train,x_test = load_images()
    # After loading train and evaluate classifier.
    
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.export_autokeras_model('best_auto_keras_model.h5')
    predictions = clf.predict(x_test)
    print(predictions)

    #clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    #y = clf.evaluate(x_test, y_test)
    #print(y * 100)
    

def run_audanet():

    x_train, y_train,x_test = load_images()
    # Define the model head for computing loss and evaluation metrics.
    head = MultiClassHead(n_classes=10)

    # Feature columns define how to process examples.
    feature_columns = ...

    # Learn to ensemble linear and neural network models.
    estimator = adanet.AutoEnsembleEstimator(
        head=head,
        candidate_pool={
            "linear":
                tf.estimator.LinearEstimator(
                    head=head,
                    feature_columns=feature_columns,
                    optimizer=...),
            "dnn":
                tf.estimator.DNNEstimator(
                    head=head,
                    feature_columns=feature_columns,
                    optimizer=...,
                    hidden_units=[1000, 500, 100])},
        max_iteration_steps=50)

    estimator.train(input_fn=x_train, steps=100)
    metrics = estimator.evaluate(input_fn=y_train)
    predictions = estimator.predict(input_fn=x_test)

if __name__ == '__main__':
    run_audanet()
