# To run first unzip ../load_raw_image_data.zip into ../
# so the train and test directories reside in this directory 

from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset
import numpy as np
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

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense

def cnn_model(features, labels, mode, params):
    images = list(features.values())[0] # get values from dict
    
    x = tf.keras.layers.Conv2D(32,
                               kernel_size=7,
                               activation='relu')(images)
    x = tf.keras.layers.MaxPooling2D(strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    logits = tf.keras.layers.Dense(10)(x)

def run_audanet():

    x_train, y_train,x_test = load_images()
    x_train = x_train / 255 # map values between 0 and 1
    x_test  = x_test / 255  # map values between 0 and 1

    x_train = x_train.astype(np.float32) # cast values to float32
    x_test = x_test.astype(np.float32)   # cast values to float32

    y_train = y_train.astype(np.int32) # cast values to int32
    y_train = y_train.astype(np.int32)   # cast values to int32

    EPOCHS = 10
    BATCH_SIZE = 32
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=False)

    adanet_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        batch_size=BATCH_SIZE,
        num_epochs=1,
        shuffle=False)

    classifier = tf.estimator.Estimator(model_fn=cnn_model)

    results, _ = tf.estimator.train_and_evaluate(classifier,
    train_spec=tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=EPOCHS),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=test_input_fn,
        steps=None))
    print("Accuracy:", results["accuracy"])
    print("Loss:", results["loss"])

if __name__ == '__main__':
    run_audanet()
