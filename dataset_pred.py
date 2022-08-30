#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('/home/argha/Documents/driver-head-pose/final_anirban_df.csv')


# In[3]:


df


# In[4]:


plt.figure(figsize=(14,8))
df['level'].hist()


# In[5]:


rp_y, doppz = [], []

for r_y, d_z in zip(df['rp_y'], df['doppz']):
    rp_y.append(ast.literal_eval(r_y))
    doppz.append(ast.literal_eval(d_z))


# In[6]:


doppz = np.array(doppz)


# In[7]:


plt.figure(figsize=(14,8))
sns.heatmap(doppz[:,:,8:18].mean(axis=0))
plt.show()
plt.figure(figsize=(14,8))
sns.heatmap(doppz[:,:,8:18].std(axis=0))


# In[8]:


xlim0,xlim1=400,700

fig=plt.figure(figsize=(12,7))
ax=fig.add_subplot(311)
map_dict=dict(zip(['looking_forward', 'looking_down', 'looking_left', 'looking_right',
       'looking_up', 'looking_up_and_left', 'looking_up_and_right'],['k', 'orange', 'blue', 'cyan', 'red',
                                                                     'green', 'yellow']))
labels=df['level'].map(map_dict)
ax.scatter(range(doppz.shape[0]),doppz[:,:,8:18].reshape(-1,160).sum(axis=1),c=labels)
ax.plot(range(doppz.shape[0]),doppz[:,:,8:18].reshape(-1,160).sum(axis=1),linestyle='--',c='k')
ax.set_xlim(xlim0,xlim1)

pathes = [mpatches.Patch(color=c, label=v) for v,c in map_dict.items()]
ax.legend(handles=pathes,ncol=3,bbox_to_anchor=(0.3, 1.05))

ax=fig.add_subplot(312)
ax.plot(abs(df['acc'] - 10.518497809158944))
ax.set_xlim(xlim0,xlim1)

ax=fig.add_subplot(313)
ax.plot(df['y'])
ax.plot(df['x'])

ax.set_xlim(xlim0,xlim1)


# In[9]:


norm=lambda e: (e-e.mean())/e.std()


# In[10]:


sns.regplot(norm(abs(df['acc'] - 10.518497809158944)),
            norm(doppz[:,:,8:18].reshape(-1,160).sum(axis=1)))


# In[11]:


sns.regplot(norm(df['y']),
            norm(doppz[:,:,8:18].reshape(-1,160).sum(axis=1)))


# In[12]:


X=doppz[:,:,8:18].reshape(-1,160)


# In[13]:


Y=df['level'].values


# In[14]:


rf=RandomForestClassifier()


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[16]:


rf.fit(X_train,y_train)


# In[17]:


pred=rf.predict(X_train)


# In[18]:


print(classification_report(y_train,pred))


# In[19]:


pred=rf.predict(X_test)


# In[20]:


print(classification_report(y_test,pred))


# In[ ]:




