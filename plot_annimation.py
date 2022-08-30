#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import glob
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


def read_data(f):
    df=pd.read_csv(f)
    df['user']=f.split("_")[1]
    return df


# In[3]:


df = pd.concat([read_data(f) for f in glob.glob("./data/*")]).reset_index(drop=False)


# In[4]:


doppz = []

for d_z in df['doppz']:
    doppz.append(ast.literal_eval(d_z))
doppz = np.array(doppz)


# In[5]:


fig, axs = plt.subplots(1,2,figsize=(9,7), gridspec_kw={'width_ratios': [8, 1]})
fig.show()


# In[7]:


for i in range(doppz.shape[0]):
    axs[0].cla();axs[1].cla()
    axs[0].set_title(df['activity'][i])
    sns.heatmap(doppz[i],ax=axs[0],cbar_ax=axs[1])
    fig.canvas.draw()


# In[8]:


def plot_relation(df,xlim0=100,xlim1=700):
    doppz = []

    for d_z in df['doppz']:
        doppz.append(ast.literal_eval(d_z))
    doppz = np.array(doppz)
    
    fig=plt.figure(figsize=(12,7))
    ax=fig.add_subplot(311)
    map_dict=dict(zip(['looking forward', 'Talking', 'yawning', 'looking right', 'looking left', 'looking up'],
                      ['k', 'orange', 'blue', 'cyan', 'red','green']))
    labels=df['activity'].map(map_dict)
    ax.scatter(range(doppz.shape[0]),doppz[:,:,0:9].reshape(-1,128*9).sum(axis=1),c=labels)
    ax.plot(range(doppz.shape[0]),doppz[:,:,0:9].reshape(-1,128*9).sum(axis=1),linestyle='--',c='k')
    ax.set_xlim(xlim0,xlim1)

    pathes = [mpatches.Patch(color=c, label=v) for v,c in map_dict.items()]
    ax.legend(handles=pathes,ncol=3,bbox_to_anchor=(0.3, 1.05))

    ax=fig.add_subplot(312)
    ax.plot(df['mar'].values, label='mar')
    ax.set_xlim(xlim0,xlim1)
    ax.legend()

    ax=fig.add_subplot(313)
    ax.plot(df['y'].values, label='y')
    ax.plot(df['x'].values, label='x')
    ax.legend()
    ax.set_xlim(xlim0,xlim1)


# In[ ]:


plot_relation(df,0,len(df))


# In[ ]:




