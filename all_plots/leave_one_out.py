import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

static_users = ['argha', 'anirban', 'bishakh', 'aritra']
models_types = ['rf', 'vgg16', 'cnn']

all_f1 = []
users = []

i = 0
for user in static_users:
    user_f1 = []
    users.append(i)
    i += 1
    for mdl in models_types:
        for file in glob.glob(f'../models/results/static_{user}_lo*_{mdl}*.pickle'):
            with open(file, "rb") as f:
                data = pickle.load(f)
                user_f1.append(data['f1'])
    all_f1.append(user_f1)

N = 4
ind = np.arange(N)
width = 0.25
bars = []
cmap = ['r', 'g', 'b', 'm']

for j in range(len(models_types)):
    for i in range(len(users)):
        bar = all_f1[i][j]
        bar1 = plt.bar(ind + width * j, bar, width, color=cmap[j])
        bars.append(bar1)

plt.show()
