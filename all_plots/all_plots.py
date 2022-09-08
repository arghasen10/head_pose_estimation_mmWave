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

time_then = input('Engter date time: ')


def plot_cfm():
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
            plt.yticks(rotation=25)
            plt.xticks(rotation=25)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()


def plot_bar_f1():
    merged_f1_score_arr = []
    for f_cfm in glob.glob(f'../models/results/merged{time_then}*.pickle'):
        with open(f_cfm, "rb") as f:
            data = pickle.load(f)
            merged_f1_score_arr.append(data['f1'])

    static_f1_score_arr = []
    for f_cfm in glob.glob(f'../models/results/static{time_then}*.pickle'):
        with open(f_cfm, "rb") as f:
            data = pickle.load(f)
            static_f1_score_arr.append(data['f1'])

    driving_f1_score_arr = []
    model_type = []
    for f_cfm in glob.glob(f'../models/results/driving{time_then}*.pickle'):
        model_type.append(f_cfm.split('.')[-2].split('_')[-2])
        with open(f_cfm, "rb") as f:
            data = pickle.load(f)
            driving_f1_score_arr.append(data['f1'])

    labels = ['Merged', 'Static', 'Driving']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, merged_f1_score_arr, width, label='RF')
    rects2 = ax.bar(x, static_f1_score_arr, width, label='VGG16')
    rects3 = ax.bar(x + width, driving_f1_score_arr, width, label='2D-CNN')
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


def leave_one_out():
    static_users = ['argha', 'anirban', 'bishakh', 'aritra']
    models_types = ['rf', 'vgg16', 'cnn']

    all_f1 = []

    for user in static_users:
        user_f1 = []
        for mdl in models_types:
            for file in glob.glob(f'../models/results/static_{user}_lo*{time_then}_{mdl}*.pickle'):
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


def personalized_plot():
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
            for file in glob.glob(f'../models/results/static_{user}_per{time_then}_{mdl}*.pickle'):
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


plot_cfm()
plot_bar_f1()
leave_one_out()
personalized_plot()
