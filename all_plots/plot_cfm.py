import matplotlib.pyplot as plt
import pickle
import glob
import seaborn as sns
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

labels = ['looking\nforward', 'Talking', 'yawning', 'looking\nright', 'looking\nleft']
cfm_ps = []
for f_cfm in glob.glob('../models/results/merged*.pickle'):
    with open(f_cfm, "rb") as f:
        data = pickle.load(f)
        df_cm = pd.DataFrame(np.array(data['conf_mat']), index=[i for i in labels], columns=[i for i in labels])
        sns.heatmap(df_cm, annot=True, cmap="Blues")
        plt.yticks(rotation=25)
        plt.xticks(rotation=25)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()



