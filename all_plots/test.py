import matplotlib.pyplot as plt
import pickle
import glob
import seaborn as sns
import pandas as pd
import numpy as np

# plt.rcParams.update({'font.size': 20})
# plt.rcParams["figure.figsize"] = (10, 8)
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

time_then = input('Engter date time: ')


labels = ['looking\nforward', 'Talking', 'yawning', 'looking\nright', 'looking\nleft']
cfm_ps = []
for f_cfm in glob.glob(f'../models/results/merged{time_then}*.pickle'):
    print(f_cfm.split('_')[-2])
    with open(f_cfm, "rb") as f:
        data = pickle.load(f)
        cfm = np.array(data['conf_mat'])
        total = cfm / cfm.sum(axis=1).reshape(-1, 1)
        df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
        sns.heatmap(df_cm, annot=True, cmap="Blues")
        plt.show()
