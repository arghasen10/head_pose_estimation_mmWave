#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from library.helper import load_dataset


# In[2]:


df=load_dataset()


# In[3]:


class_map={
 'looking forward': 0,
 'looking right': 1,
 'looking left': 2,
 'yawning': 3,
 'looking up': 4,
 'Talking': 5}

df['activity']=df.activity.map(class_map)


# In[4]:


features=['doppz']
label='activity'


# In[5]:


from sklearn.ensemble import RandomForestClassifier


# In[6]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score,classification_report


# In[7]:


kf=KFold(n_splits=3,shuffle=True,random_state=42)


# In[8]:


extract_flat_feat=lambda ddf:np.array(ddf[features].values.tolist())[:,:,:,:9].reshape(-1,128*9)


# In[9]:


def evaluate(model_fn,process_fn=None):
    train_f1=[]
    test_f1=[]
    for train_idx,test_idx in kf.split(df):
        train_df=df.iloc[train_idx].copy()
        test_df=df.iloc[test_idx].copy()

        X_train=process_fn(train_df)
        y_train=train_df['activity']

        X_test=process_fn(test_df)
        y_test=test_df['activity']

        scaler=MinMaxScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
        
        model=model_fn()
        model.fit(X_train,y_train)

        pred=model.predict(X_train)
        #print(classification_report(y_train,pred))
        train_f1.append(f1_score(y_train,pred,average='weighted'))

        pred=model.predict(X_test)
        #print(classification_report(y_test,pred))
        test_f1.append(f1_score(y_test,pred,average='weighted'))

    plt.boxplot(x=[train_f1,test_f1],positions=[0,1])
    plt.show()


# In[10]:


rf_model_fn=lambda: RandomForestClassifier(n_estimators=30,criterion='gini',max_depth=20)
evaluate(rf_model_fn,extract_flat_feat)


# In[11]:


#CNN


# In[10]:


import tensorflow as tf


# In[10]:


class My2dCNN:
    def __init__(self):
        self.history=None
        self.model=self.build_network()
        self.model.summary()
    
    def build_network(self):
        cnn=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (8,9), (3,1), activation='relu', padding="valid", input_shape=(128,9,1)),
            tf.keras.layers.Conv2D(96, (5,1), (3,1), activation='relu', padding="valid"),
            tf.keras.layers.Conv2D(128, (5,1), (2,1), activation='relu', padding="valid"),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(6,activation='softmax')
        ],name='2dCNN')
        cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return cnn
    
    def fit(self,X,y,epoch=400,batchsize=32,vb=1):
        self.history=self.model.fit(X,y,epochs=epoch,batch_size=batchsize,verbose=vb)
        self.plot()
    
    def predict(self,X):
        return self.model.predict_classes(X)

    def plot(self):
        plt.plot(self.history.history['accuracy'])
        plt.show()


# In[11]:


21def evaluate(model_fn,process_fn=None):
    train_f1=[]
    test_f1=[]
    for train_idx,test_idx in kf.split(df):
        train_df=df.iloc[train_idx].copy()
        test_df=df.iloc[test_idx].copy()


        X_train=process_fn(train_df)
        y_train=train_df['activity']

        X_test=process_fn(test_df)
        y_test=test_df['activity']
        
        max_val=X_train.max()
        X_train=X_train/max_val
        X_test=X_test/max_val
        
        model=model_fn()
        model.fit(X_train,y_train)

        pred=model.predict(X_train)
        #print(classification_report(y_train,pred))
        train_f1.append(f1_score(y_train,pred,average='weighted'))

        pred=model.predict(X_test)
        #print(classification_report(y_test,pred))
        test_f1.append(f1_score(y_test,pred,average='weighted'))

    plt.boxplot(x=[train_f1,test_f1],positions=[0,1])
    plt.show()


# In[41]:


extrat_2dcnn_feat=lambda df:np.array(df[features].values.tolist())[:,:,:,:9].transpose(0,2,3,1)
cnn2d_model_fn=lambda : My2dCNN()
evaluate(cnn2d_model_fn,extrat_2dcnn_feat)


# In[12]:


class My2dCNN_LSTM:
    def __init__(self):
        self.history=None
        self.model=self.build_network()
        self.model.summary()
    
    def build_network(self):
        cnn=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (8,9), (3,1), activation='relu', padding="valid", input_shape=(6,128,9,1)),
            tf.keras.layers.Conv2D(96, (5,1), (3,1), activation='relu', padding="valid"),
            tf.keras.layers.Conv2D(128, (5,1), (2,1), activation='relu', padding="valid"),
            tf.keras.layers.Reshape((6,-1,128)),
            tf.keras.layers.Lambda(lambda a:tf.math.reduce_mean(a,axis=2),name='Avg'),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(6,activation='softmax')
        ],name='2dCNN')
        cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return cnn
    
    def fit(self,X,y,epoch=400,batchsize=32,vb=1):
        self.history=self.model.fit(X,y,epochs=epoch,batch_size=batchsize,verbose=vb)
        self.plot()
    
    def predict(self,X):
        return self.model.predict_classes(X)

    def plot(self):
        plt.plot(self.history.history['accuracy'])
        plt.show()


# In[13]:


def extrat_2dcnn_lstm_feat(df,window=6):
    temp=np.array(df[features].values.tolist())[:,:,:,:9].transpose(0,2,3,1)
    X=np.array([temp[i:i+window] for i in range(temp.shape[0]-window)])
    y=df[label][window:].values
    return X,y

cnn_lstm_model_fn=lambda : My2dCNN_LSTM()


# In[15]:


def evaluate(model_fn,window=6):
    train_f1=[]
    test_f1=[]
    data,label=extrat_2dcnn_lstm_feat(df,window)
    for train_idx,test_idx in kf.split(data):

        X_train=data[train_idx]
        y_train=label[train_idx]

        X_test=data[test_idx]
        y_test=label[test_idx]
        
        max_val=X_train.max()
        X_train=X_train/max_val
        X_test=X_test/max_val
        
        model=model_fn()
        model.fit(X_train,y_train)

        pred=model.predict(X_train)
        #print(classification_report(y_train,pred))
        train_f1.append(f1_score(y_train,pred,average='weighted'))

        pred=model.predict(X_test)
        #print(classification_report(y_test,pred))
        test_f1.append(f1_score(y_test,pred,average='weighted'))

    plt.boxplot(x=[train_f1,test_f1],positions=[0,1])
    plt.show()


# In[16]:


evaluate(cnn_lstm_model_fn,window=6)


# In[17]:


#########
#2d CNN
class My2dCNN_ch:
    def __init__(self):
        self.history=None
        self.model=self.build_network()
        self.model.summary()
    
    def build_network(self):
        cnn=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (8,9), (3,1), activation='relu', padding="valid", input_shape=(128,9,6)),
            tf.keras.layers.Conv2D(96, (5,1), (3,1), activation='relu', padding="valid"),
            tf.keras.layers.Conv2D(128, (5,1), (2,1), activation='relu', padding="valid"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64,'relu'),
            tf.keras.layers.Dense(6,activation='softmax')
        ],name='2dCNN')
        cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return cnn
    
    def fit(self,X,y,epoch=400,batchsize=32,vb=1):
        self.history=self.model.fit(X,y,epochs=epoch,batch_size=batchsize,verbose=vb)
        self.plot()
    
    def predict(self,X):
        return self.model.predict_classes(X)

    def plot(self):
        plt.plot(self.history.history['accuracy'])
        plt.show()


# In[18]:


My2dCNN_ch()


# In[19]:


def extrat_2dcnn_ch_feat(df,window=6):
    temp=np.array(df[features[0]].values.tolist())[:,:,:9]
    X=np.array([temp[i:i+window] for i in range(temp.shape[0]-window)]).transpose(0,2,3,1)
    y=df[label][window:].values
    return X,y

cnn_ch_model_fn=lambda : My2dCNN_ch()


# In[20]:


def evaluate(model_fn,window=6):
    train_f1=[]
    test_f1=[]
    data,label=extrat_2dcnn_ch_feat(df,window)
    for train_idx,test_idx in kf.split(data):

        X_train=data[train_idx]
        y_train=label[train_idx]

        X_test=data[test_idx]
        y_test=label[test_idx]
        
        max_val=X_train.max()
        X_train=X_train/max_val
        X_test=X_test/max_val
        
        model=model_fn()
        model.fit(X_train,y_train)

        pred=model.predict(X_train)
        #print(classification_report(y_train,pred))
        train_f1.append(f1_score(y_train,pred,average='weighted'))

        pred=model.predict(X_test)
        #print(classification_report(y_test,pred))
        test_f1.append(f1_score(y_test,pred,average='weighted'))

    plt.boxplot(x=[train_f1,test_f1],positions=[0,1])
    plt.show()


# In[21]:


evaluate(cnn_ch_model_fn)


# In[ ]:





# In[ ]:




