import glob
import numpy as np
import pandas as pd

def read_data(f):
    df=pd.read_csv(f)
    df['user']=f.split("_")[1]
    return df

def modified_glob(pattern, exclude_substr):
    files=[f for f in glob.glob(pattern) if exclude_substr not in f]
    print(files)
    return files

def load_dataset(loc="./data/*!a"):
    temp=loc.split("!")
    if len(temp)==1:
        pattern=temp[0]
        exclude_substr='@#'
    else:
        pattern,exclude_substr=temp
    df=pd.concat([read_data(f) for f in modified_glob(pattern,exclude_substr)])
    df['doppz']=df['doppz'].apply(lambda e:eval(e))
    return df[['user','doppz','activity']].reset_index(drop=True)