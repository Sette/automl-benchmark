from autokeras.utils import pickle_from_file
from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset

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
    

x_train, y_train,x_test = load_images()
# After loading train and evaluate classifier.

model = pickle_from_file('best_auto_keras_model.h5')
predictions = model.predict(x_test)
error_out = open('predict_auto_keras',"w")
error_out.write(str(predictions))
error_out.close() 
print(predictions)