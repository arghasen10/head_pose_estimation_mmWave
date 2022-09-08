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
        for file in glob.glob(f'../models/results/static_{user}_per*_{mdl}*.pickle'):
            with open(file, "rb") as f:
                data = pickle.load(f)
                user_f1.append(data['f1'])
    all_f1.append(user_f1)

rf_p = []
vgg_p = []
cnn_p = []
for i in range(len(all_f1)):
    rf_p.append(all_f1[i][0])
    vgg_p.append(all_f1[i][1])
    cnn_p.append(all_f1[i][2])

labels = ['S1', 'S2', 'S3', 'S4']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, rf_p, width, label='RF')
rects2 = ax.bar(x, vgg_p, width, label='VGG16')
rects3 = ax.bar(x + width, cnn_p, width, label='2D-CNN')

ax.set_ylabel('F1-Score')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.show()