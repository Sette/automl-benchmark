# To run first unzip ../load_raw_image_data.zip into ../
# so the train and test directories reside in this directory 

from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
import numpy as np
import adanet
import tensorflow as tf
from keras.datasets import mnist

class CNNBuilder(adanet.subnetwork.Builder):
    def __init__(self, n_convs):
        self._n_convs = n_convs
        
    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        
        images = list(features.values())[0]
        x = images
    
        for i in range(self._n_convs):
            x = Conv2D(32, kernel_size=7, activation='relu')(x)
            x = MaxPooling2D(strides=2)(x)
        
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        
        logits = Dense(10)(x)

        complexity = tf.constant(1)

        persisted_tensors = {'n_convs': tf.constant(self._n_convs)}
        
        return adanet.Subnetwork(
            last_layer=x,
            logits=logits,
            complexity=complexity,
            persisted_tensors=persisted_tensors)
    
    def build_subnetwork_train_op(self,
                                  subnetwork,
                                  loss,
                                  var_list,
                                  labels,
                                  iteration_step,
                                  summary,
                                  previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,
                                              decay=0.0)
        # NOTE: The `adanet.Estimator` increments the global step.
        return optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self,
                                       loss,
                                       var_list,
                                       logits,
                                       labels,
                                       iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""
        return tf.no_op("mixture_weights_train_op")

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        return f'cnn_{self._n_convs}'

class CNNGenerator(adanet.subnetwork.Generator):    
    def __init__(self):
        self._cnn_builder_fn = CNNBuilder    
    def generate_candidates(self,
                            previous_ensemble,
                            iteration_number,
                            previous_ensemble_reports,
                            all_reports):

        n_convs = 0
        if previous_ensemble:
            n_convs = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .persisted_tensors['n_convs'])        
        return [
            self._cnn_builder_fn(n_convs=n_convs),
            self._cnn_builder_fn(n_convs=n_convs + 1)
        ]

data_dir = "datasets/dog-breed"

def load_images():
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/labels_real.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission_real.csv",
                                        images_path=data_dir+"/test")
    print(x_test[0].shape)
    
    return x_train/255, y_train,x_test[0]/255
    


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

    EPOCHS = 10
    BATCH_SIZE = 32

    #x_train, y_train,x_test = load_images()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255 # map values between 0 and 1
    x_test  = x_test / 255  # map values between 0 and 1

    x_train = x_train.astype(np.float32) # cast values to float32
    x_test = x_test.astype(np.float32)   # cast values to float32

    labels_train = labels_train.astype(np.int32) # cast values to int32
    labels_test = labels_test.astype(np.int32)   # cast values to int32
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

    head = tf.contrib.estimator.multi_class_head(10)
    estimator = adanet.Estimator(
    head=head,
    subnetwork_generator=CNNGenerator(),
    max_iteration_steps=5,
    evaluator=adanet.Evaluator(
        input_fn=adanet_input_fn,
        steps=None),
    adanet_loss_decay=.99)

    results, _ = tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=5),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=test_input_fn,
        steps=None))
    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])
    print(ensemble_architecture(results))

if __name__ == '__main__':
    run_audanet()
