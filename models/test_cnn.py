import numpy as np
from cnn_model import get_dataset,get_model
from sklearn.metrics import classification_report,confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--loc", default='../static_dataset/processed/data/*', help="Root Path to dataset")
parser.add_argument("-mp", "--model_path", default="./saved_weights/static_saved_weights_cnn.h5",help="Destination to model weights")
parser.add_argument("-c", "--class_count", default=600, help="class_count in balanced dataset")
args = parser.parse_args()


if __name__=='__main__':
    X_train, X_test, y_train, y_test=get_dataset(args)
    model=get_model()
    model.load_weights(args.model_path)
    pred=model.predict(X_test)
    pred_class=np.argmax(pred,axis=1)
    true_class=np.argmax(y_test,axis=1)
    print("report:\n",classification_report(true_class,pred_class))
    print("confusion matrix:\n",confusion_matrix(true_class,pred_class))