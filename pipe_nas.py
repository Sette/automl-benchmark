from autokeras import ImageClassifier
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
import numpy as np
import adanet
import tensorflow as tf
from keras.datasets import mnist
from load_utils import *
from devol import DEvol, GenomeHandler

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
        x = Dense(120, activation='relu')(x)
        
        logits = Dense(120)(x)

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



def run_autokeras():
    #x_train, y_train,x_test = load_plant_seedlings()
    #x_train, y_train,x_test = load_dog_breed()
    x_train,y_train,x_text = load_invasive_species()
    # After loading train and evaluate classifier.
    
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.export_autokeras_model('best_auto_keras_model_invasive.h5')
    predictions = clf.predict(x_test)
    print(predictions)

    submission = pd.DataFrame({
        "target": predictions
    })

    submission.to_csv('invasive_'+'autokeras_submission.csv', index=False)


    #clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    #y = clf.evaluate(x_test, y_test)
    #print(y * 100)


def run_adanet():

    EPOCHS = 10
    BATCH_SIZE = 32

    #x_train, y_train,x_test = load_images()
    x_train, y_train,x_test = load_dog_breed()

    y_train = pd.get_dummies(y_train,sparse = True)

    y_train = np.asarray(y_train)
        
    x_train = x_train / 255 # map values between 0 and 1
    x_test  = x_test / 255  # map values between 0 and 1

    #x_train = x_train.astype(np.float32) # cast values to float32
    #x_test = x_test.astype(np.float32)   # cast values to float32

    #y_train = y_train.astype(np.int32) # cast values to int32
    

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

    head = tf.contrib.estimator.multi_class_head(120)
    estimator = adanet.Estimator(
    head=head,
    subnetwork_generator=CNNGenerator(),
    max_iteration_steps=200,
    evaluator=adanet.Evaluator(
            input_fn=adanet_input_fn,
            steps=None),
        adanet_loss_decay=.99)

    results, _ = tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=200),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=train_input_fn,
        steps=None))

    
    predictions = estimator.predict(input_fn=test_input_fn)
    
    preds = list()
    for i, val in enumerate(predictions):
        predicted_class = val['class_ids'][0]
        print(val['probabilities'])
        preds.append(predicted_class)
        prediction_confidence = val['probabilities'][predicted_class] * 100

    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])

    id_test = pd.read_csv("datasets/aerial-cactus/sample_submission.csv")

    submission = pd.DataFrame({"id": id_test.id.values, "has_cactus": preds})

    submission.to_csv("invasive_adanet_submission.csv", index=False)


def run_devol():
	X_train, y_train,X_test = load_dog_breed()
	genome_handler = GenomeHandler(max_conv_layers=6, 
                               max_dense_layers=2, # includes final dense layer
                               max_filters=256,
                               max_dense_nodes=1024,
                               input_shape=X_train.shape[1:],
                               n_classes=120)
	devol = DEvol(genome_handler)
	dataset = ((X_train,y_train),(X_train))
	model = devol.run(dataset=dataset,
                  num_generations=20,
                  pop_size=20,
                  epochs=5)
	print(model.summary())

if __name__ == '__main__':
    run_autokeras()
