import sys
import pickle
sys.path.append('../')
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from library.augumentation import Dataset
from sklearn.model_selection import train_test_split

def convert_to_neumeric(label):
    lbl_map = \
    {'looking forward': 0,
     'Talking': 1,
     'yawning': 2,
     'looking right': 3,
     'looking left': 4,
     'looking up': 0}
    return np.array(list(map(lambda e: lbl_map[e], label)))

def split_dataset(data,label):
    np.random.seed(101)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def read_mydata(train_pattern=None, test_pattern=None, class_count=600):
    if test_pattern is None:
        dataset = Dataset(loc=train_pattern,class_count=class_count,frame_stack=4,dop_min=1375,dop_max=5293)
        data=dataset.data;label=convert_to_neumeric(dataset.label)
        return split_dataset(data,label)
    else:
        dataset = Dataset(loc=train_pattern,class_count=class_count,frame_stack=4,dop_min=1375,dop_max=5293)
        X_train=dataset.data;y_train=convert_to_neumeric(dataset.label)
        
        dataset = Dataset(loc=test_pattern,class_count=class_count,frame_stack=4,dop_min=1375,dop_max=5293)
        X_test=dataset.data;y_test=convert_to_neumeric(dataset.label)
        return X_train, X_test, y_train, y_test

class Model:
    def __init__(self,X_train, X_test, y_train, y_test,model_class):
        self.model=model_class(X_train, X_test, y_train, y_test)

    def train(self,save_path=None):
        self.model.train(save_path)

    def test(self,identifier=None):
        self.model.test(identifier)

class rf_model:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test=self.rf_process(X_train, X_test, y_train, y_test)
        self.model=self.get_rf_model()

    def PoolOp(self,img, ksize=(16, 16), op=np.mean):
        i_h, i_w, i_c = img.shape
        k_h, k_w = ksize
        row = []
        for c in range(i_c):
            for i in range(i_h // k_h):
                for j in range(i_w // k_w):
                    row.append(op(img[k_h * i:k_h * i + k_h, k_w * j:k_w * j + k_w, c]))
        return np.array(row)

    def apply_pool(self,v):
        return np.concatenate([self.PoolOp(v, op=np.min),
                               self.PoolOp(v, op=np.max),
                               self.PoolOp(v, op=np.mean),
                               self.PoolOp(v, op=np.std)]).tolist()

    def rf_process(self,X_train, X_test, y_train, y_test):
        return (np.array([self.apply_pool(d) for d in X_train]),
            np.array([self.apply_pool(d) for d in X_test]),
            y_train,
            y_test)

    def get_rf_model(self):
        rf = RandomForestClassifier(random_state=101)
        return rf

    def train(self,save_path=None):
        self.model.fit(self.X_train,self.y_train)
    
    def test(self,identifier=None):
        pred=self.model.predict(self.X_test)
        conf_matrix=confusion_matrix(self.y_test,pred)
        class_report=classification_report(self.y_test,pred)
        f1=f1_score(self.y_test,pred,average="weighted")
        result="confusion matrix\n"+repr(conf_matrix)+"\n"+"report\n"+class_report+"\nf1_score(weighted)\n"+repr(f1)
        pickle_log={'conf_mat':conf_matrix,'f1':f1}
        with open(f"./results/{identifier}_rf_model.pickle","wb") as f:
            pickle.dump(pickle_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"./results/{identifier}_rf_model.txt","w+") as f:
            f.write(result)
        print(result)


class vgg16_model:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test=self.vgg16_process(X_train, X_test, y_train, y_test)
        self.model=self.get_vgg16_model()

    def vgg16_process(self,X_train, X_test, y_train, y_test):
        return (preprocess_input(np.uint8(X_train[:,:,:,-3:]*255)),
                preprocess_input(np.uint8(X_test[:,:,:,-3:]*255)), y_train, y_test)

    def get_vgg16_model(self):
        tf.random.set_seed(101)
        vgg16_topless = VGG16(weights="./vgg16_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(48,48,3))
        vgg16_topless.trainable = False
        model=tf.keras.Sequential([
            vgg16_topless,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32,"relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(5,"softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics="accuracy")
        return model

    def train(self,save_path=None):
        best_save=tf.keras.callbacks.ModelCheckpoint(filepath=save_path,save_weights_only=True,
                                                    monitor='val_accuracy',mode='max',save_best_only=True)
        self.model.fit(self.X_train,self.y_train,
                    epochs=2000,
                    validation_split=0.2,
                    batch_size=32,
                    callbacks=[best_save])

    def test(self,identifier=None):
        pred=np.argmax(self.model.predict(self.X_test),axis=1)
        conf_matrix=confusion_matrix(self.y_test,pred)
        class_report=classification_report(self.y_test,pred)
        f1=f1_score(self.y_test,pred,average="weighted")
        result="confusion matrix\n"+repr(conf_matrix)+"\n"+"report\n"+class_report+"\nf1_score(weighted)\n"+repr(f1)
        pickle_log={'conf_mat':conf_matrix,'f1':f1}
        with open(f"./results/{identifier}_vgg16_model.pickle","wb") as f:
            pickle.dump(pickle_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"./results/{identifier}_vgg16_model.txt","w+") as f:
            f.write(result)
        print(result)


class cnn_model:
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test=self.cnn_process(X_train, X_test, y_train, y_test)
        self.model=self.get_cnn_model()

    def cnn_process(self,X_train, X_test, y_train, y_test):
        return X_train, X_test, y_train, y_test

    def get_cnn_model(self):
        tf.random.set_seed(101)
        model=tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,(5,5),(1,1),padding="same",activation='relu',input_shape=(48,48,4)),
            tf.keras.layers.MaxPool2D((3,3)),
            tf.keras.layers.Conv2D(64,(3,3),(1,1),padding="same",activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(96,(3,3),(1,1),padding="same",activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32,"relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(5,"softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics="accuracy")
        return model

    def train(self,save_path=None):
        best_save=tf.keras.callbacks.ModelCheckpoint(filepath=save_path,save_weights_only=True,
                                                    monitor='val_accuracy',mode='max',save_best_only=True)
        self.model.fit(self.X_train,self.y_train,
                    epochs=2000,
                    validation_split=0.2,
                    batch_size=32,
                    callbacks=[best_save])

    def test(self,identifier=None):
        pred=np.argmax(self.model.predict(self.X_test),axis=1)
        conf_matrix=confusion_matrix(self.y_test,pred)
        class_report=classification_report(self.y_test,pred)
        f1=f1_score(self.y_test,pred,average="weighted")
        result="confusion matrix\n"+repr(conf_matrix)+"\n"+"report\n"+class_report+"\nf1_score(weighted)\n"+repr(f1)
        pickle_log={'conf_mat':conf_matrix,'f1':f1}
        with open(f"./results/{identifier}_cnn_model.pickle","wb") as f:
            pickle.dump(pickle_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"./results/{identifier}_cnn_model.txt","w+") as f:
            f.write(result)
        print(result)