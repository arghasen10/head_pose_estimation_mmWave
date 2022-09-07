import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def plot_relation(df, xlim0=100, xlim1=700):
    doppz = np.array(df['doppz'].values.tolist())
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(311)
    map_dict = dict(zip(['looking forward', 'Talking', 'yawning', 'looking right', 'looking left'],
                        ['k', 'orange', 'blue', 'cyan', 'red', 'green']))
    labels = df['activity'].map(map_dict)
    print(labels)
    sum_val = doppz.reshape(-1, 128 * 64).mean(axis=1)
    ax.scatter(range(len(sum_val)), sum_val, c=labels, label='mean')
    ax.legend()
    ax.plot(range(len(sum_val)), sum_val, linestyle='--', c='k', label='mean')
    ax.set_xlim(xlim0, xlim1)
    ax.legend()

    pathes = [mpatches.Patch(color=c, label=v) for v, c in map_dict.items()]
    ax.legend(handles=pathes, ncol=3, bbox_to_anchor=(0.2, 1.05))

    ax = fig.add_subplot(312)
    ax.plot(df['mar'].values, label='mouth feature')
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(-10, 150)
    ax.legend()

    ax = fig.add_subplot(313)
    ax.plot(df['y'].values, color='m', label='head orientation')
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(-15, 35)
    ax.legend()
    plt.show()


df = pd.read_csv('../static_dataset/processed/denoised/final_argha_df.csv')
df = df[df.activity != 'looking up']
df['doppz'] = df['doppz'].apply(lambda e: eval(e))
plot_relation(df, xlim0=183, xlim1=500)
