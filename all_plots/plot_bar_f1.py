import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

labels = ['looking\nforward', 'Talking', 'yawning', 'looking\nright', 'looking\nleft']
merged_f1_score_arr = []
for f_cfm in glob.glob('../models/results/merged*.pickle'):
    with open(f_cfm, "rb") as f:
        data = pickle.load(f)
        merged_f1_score_arr.append(data['f1'])

static_f1_score_arr = []
for f_cfm in glob.glob('../models/results/static2*.pickle'):
    with open(f_cfm, "rb") as f:
        data = pickle.load(f)
        static_f1_score_arr.append(data['f1'])

driving_f1_score_arr = []
model_type = []
for f_cfm in glob.glob('../models/results/driving2*.pickle'):
    model_type.append(f_cfm.split('.')[-2].split('_')[-2])
    with open(f_cfm, "rb") as f:
        data = pickle.load(f)
        driving_f1_score_arr.append(data['f1'])

N = 3
ind = np.arange(N)
width = 0.25
bars = []
cmap = ['r', 'g', 'b']
for i in range(3):
    bar = [merged_f1_score_arr[i], static_f1_score_arr[i], driving_f1_score_arr[i]]
    bar1 = plt.bar(ind + width * i, bar, width, color=cmap[i])
    bars.append(bar1)

plt.xlabel("Dataset")
plt.ylabel('F1 Score')

plt.xticks(ind + width, ['Merged', 'Static', 'Driving'])
plt.legend(bars, model_type)
plt.show()
