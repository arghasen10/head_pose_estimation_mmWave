import glob
import numpy as np
import pandas as pd

def read_data(f):
    df=pd.read_csv(f)
    df['user']=f.split("_")[1]
    return df

def load_dataset(loc="./data/*"):
    df=pd.concat([read_data(f) for f in glob.glob(loc)])
    df['doppz']=df['doppz'].apply(lambda e:eval(e))
    return df[['user','doppz','activity']].reset_index(drop=True)