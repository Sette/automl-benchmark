# To run first unzip ../load_raw_image_data.zip into ../
# so the train and test directories reside in this directory 

from autokeras import ImageClassifier
from autokeras.image.image_supervised import load_image_dataset


data_dir = "datasets/dog-breed"

def load_images():
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/labels_real.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission_real.csv",
                                        images_path=data_dir+"/train")
    print(x_test.shape)
    
    return x_train, y_train,x_test
    


def run():
    x_train, y_train, x_test = load_images()
    # After loading train and evaluate classifier.
    
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    #clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    #y = clf.evaluate(x_test, y_test)
    #print(y * 100)
    


if __name__ == '__main__':
    run()
