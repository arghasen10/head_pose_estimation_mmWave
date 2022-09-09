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
labels_4 = ['looking\nforward', 'yawning', 'looking\nright', 'looking\nleft']

static_RF = [[112, 29, 31, 17, 4], [45, 111, 15, 11, 3], [24, 17, 135, 8, 7], [10, 5, 6, 134, 3], [14, 7, 13, 7, 132]]
driving_rf = [[116, 35, 18, 15], [54, 96, 15, 12], [37, 13, 122, 9], [42, 20, 2, 114]]
static_cnn = [[190, 30, 40, 24, 16], [47, 229, 9, 6, 6], [36, 14, 250, 4, 4], [16, 14, 7, 263, 5], [8, 14, 2, 1, 265]]
driving_cnn = [[167, 47, 35, 57], [68, 172, 7, 37], [68, 7, 226, 18], [69, 28, 20, 174]]
static_vgg = [[65, 32, 62, 17, 17], [12, 118, 33, 11, 11], [13, 20, 136, 14, 8], [7, 9, 17, 115, 10],
              [15, 11, 18, 7, 122]]
driving_vgg = [[80, 43, 27, 34], [33, 109, 23, 12], [24, 18, 123, 16], [40, 23, 8, 107]]

f1_rf = [0.7, 0.63]
f1_cnn = [0.82, 0.65]
f1_vgg = [0.61, 0.59]

static_RF_activity = [0.56, 0.63, 0.69, 0.8, 0.82]
static_cnn_activity = [0.66, 0.79, 0.83, 0.89, 0.92]
static_vgg_activity = [0.43, 0.61, 0.6, 0.71, 0.72]

driving_RF_activity = [0.54, 0.56, 0.72, 0.7]
driving_cnn_activity = [0.51, 0.66, 0.76, 0.62]
driving_vgg_activity = [0.44, 0.59, 0.68, 0.62]

cfm = np.array(driving_vgg)
final_label = labels if cfm.shape[0] == 5 else labels_4

total = cfm / cfm.sum(axis=1).reshape(-1, 1)
df_cm = pd.DataFrame(total, index=[i for i in final_label], columns=[i for i in final_label])
ax = sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
# ax.set_ylim(0, 1)
plt.yticks(rotation=25)
plt.xticks(rotation=25)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

#Overall
labels = ['Static', 'Driving']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, f1_rf, width, label='RF')
rects2 = ax.bar(x, f1_vgg, width, label='VGG16')
rects3 = ax.bar(x + width, f1_cnn, width, label='2D-CNN')
ax.set_ylim(0, 1)
ax.set_ylabel('F1-Score')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()
plt.grid(alpha=0.2)
plt.show()

#STtaic Activity
labels = ['looking\nforward', 'Talking', 'yawning', 'looking\nright', 'looking\nleft']
labels_4 = ['looking\nforward', 'yawning', 'looking\nright', 'looking\nleft']

# final_label = labels if cfm.shape[0] == 5 else labels_4

x = np.arange(len(labels_4))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, driving_RF_activity, width, label='RF')
rects2 = ax.bar(x, driving_vgg_activity, width, label='VGG16')
rects3 = ax.bar(x + width, driving_cnn_activity, width, label='2D-CNN')
ax.set_ylim(0, 1)
ax.set_ylabel('F1-Score')
ax.set_xticks(x, labels_4)
ax.legend()

fig.tight_layout()
plt.grid(alpha=0.2)
plt.show()

# def leave_one_out():
#     static_users = ['argha', 'anirban', 'bishakh', 'aritra']
#     models_types = ['rf', 'vgg16', 'cnn']
#
#     all_f1 = []
#
#     for user in static_users:
#         user_f1 = []
#         for mdl in models_types:
#             for file in glob.glob(f'../models/results/static_{user}_lo*{time_then}_{mdl}*.pickle'):
#                 with open(file, "rb") as f:
#                     data = pickle.load(f)
#                     user_f1.append(data['f1'])
#         all_f1.append(user_f1)
#
#     rf_p = []
#     vgg_p = []
#     cnn_p = []
#     for i in range(len(all_f1)):
#         rf_p.append(all_f1[i][0])
#         vgg_p.append(all_f1[i][1])
#         cnn_p.append(all_f1[i][2])
#
#     labels = ['S1', 'S2', 'S3', 'S4']
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.25  # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, rf_p, width, label='RF')
#     rects2 = ax.bar(x, vgg_p, width, label='VGG16')
#     rects3 = ax.bar(x + width, cnn_p, width, label='2D-CNN')
#
#     ax.set_ylabel('F1-Score')
#     ax.set_xticks(x, labels)
#     ax.legend()
#
#     fig.tight_layout()
#
#     plt.show()
#
#
# def personalized_plot():
#     static_users = ['argha', 'anirban', 'bishakh', 'aritra']
#     models_types = ['rf', 'vgg16', 'cnn']
#
#     all_f1 = []
#     users = []
#
#     i = 0
#     for user in static_users:
#         user_f1 = []
#         users.append(i)
#         i += 1
#         for mdl in models_types:
#             for file in glob.glob(f'../models/results/static_{user}_per{time_then}_{mdl}*.pickle'):
#                 with open(file, "rb") as f:
#                     data = pickle.load(f)
#                     user_f1.append(data['f1'])
#         all_f1.append(user_f1)
#
#     rf_p = []
#     vgg_p = []
#     cnn_p = []
#     for i in range(len(all_f1)):
#         rf_p.append(all_f1[i][0])
#         vgg_p.append(all_f1[i][1])
#         cnn_p.append(all_f1[i][2])
#
#     labels = ['S1', 'S2', 'S3', 'S4']
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.25  # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, rf_p, width, label='RF')
#     rects2 = ax.bar(x, vgg_p, width, label='VGG16')
#     rects3 = ax.bar(x + width, cnn_p, width, label='2D-CNN')
#
#     ax.set_ylabel('F1-Score')
#     ax.set_xticks(x, labels)
#     ax.legend()
#
#     fig.tight_layout()
#
#     plt.show()
#
#
# plot_cfm()
# plot_bar_f1()
# leave_one_out()
# personalized_plot()
