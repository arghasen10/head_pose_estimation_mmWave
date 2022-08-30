import pandas as pd
import numpy as np
import ast
import glob
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def read_data(f):
    df = pd.read_csv(f)
    df['user'] = f.split("_")[1]
    return df


def string_to_matrix(df):
    doppz = []
    for d_z in df:
        doppz.append(ast.literal_eval(d_z))
    return doppz


def noise_detector(df):
    doppz = df['doppz'].values
    index_val = []
    for i, v in enumerate(doppz):
        v = np.array(v)
        sum_val = v.reshape(-1, 128 * 64).sum(axis=1)
        if sum_val > 25000000:
            index_val.append(i)

    fig, axs = plt.subplots(1, 2, figsize=(9, 7), gridspec_kw={'width_ratios': [8, 1]})
    fig.show()
    for i in index_val:
        axs[0].cla()
        axs[1].cla()
        axs[0].set_title(df['activity'][i])
        sns.heatmap(doppz[i], ax=axs[0], cbar_ax=axs[1])
        fig.canvas.draw()
    return index_val


def noise_removal(df):
    doppz = df['doppz'].values
    index_val = []
    prev_i = 0
    last_currect = 0
    for i, v in enumerate(doppz):
        v = np.array(v)
        sum_val = v.reshape(-1, 128 * 64).sum(axis=1)
        if sum_val > 25000000:
            doppz[i] = doppz[last_currect]
        else:
            last_currect = i
    return doppz


def plot_animation(df):
    doppz = np.array(df['doppz'].values.tolist())
    fig, axs = plt.subplots(1, 2, figsize=(9, 7), gridspec_kw={'width_ratios': [8, 1]})
    fig.show()
    for i in range(doppz.shape[0]):
        axs[0].cla()
        axs[1].cla()
        axs[0].set_title(df['activity'][i])
        sns.heatmap(doppz[i], ax=axs[0], cbar_ax=axs[1])
        fig.canvas.draw()


def plot_relation(df, xlim0=100, xlim1=700):
    doppz = np.array(df['doppz'].values.tolist())
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(311)
    map_dict = dict(zip(['looking forward', 'Talking', 'yawning', 'looking right', 'looking left', 'looking up'],
                        ['k', 'orange', 'blue', 'cyan', 'red', 'green']))
    labels = df['activity'].map(map_dict)
    print(labels)
    sum_val = doppz.reshape(-1, 128 * 64).sum(axis=1)
    ax.scatter(range(len(sum_val)), sum_val, c=labels)
    ax.plot(range(len(sum_val)), len(sum_val), linestyle='--', c='k')
    ax.set_xlim(xlim0, xlim1)

    pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
    ax.legend(handles=pathes, ncol=3, bbox_to_anchor=(0.3, 1.05))

    ax = fig.add_subplot(312)
    ax.plot(df['mar'].values, label='mar')
    ax.set_xlim(xlim0, xlim1)
    ax.legend()

    ax = fig.add_subplot(313)
    ax.plot(df['y'].values, label='y')
    ax.plot(df['x'].values, label='x')
    ax.legend()
    ax.set_xlim(xlim0, xlim1)


df = pd.concat([read_data(f) for f in glob.glob("./data/*")]).reset_index(drop=False)
df['doppz'] = string_to_matrix(df['doppz'])
df['doppz'] = noise_removal(df).tolist()
# plot_animation(df)
plot_relation(df[df.user == 'bishakh1'], 0, len(df[df.user == 'bishakh1']))
