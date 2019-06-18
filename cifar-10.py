from keras.datasets import cifar10
import autokeras as ak


# loadning cifar10 from keras
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Train shape: ", X_train.shape)

print("Train shape: ", X_test.shape)

clf = ak.ImageClassifier()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_prediction = clf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=y_prediction))