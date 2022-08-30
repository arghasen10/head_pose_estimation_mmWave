import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from library.helper import load_dataset
import ast


df = load_dataset().reset_index(drop=True)
doppz = []

for d_z in df['doppz']:
    doppz.append(ast.literal_eval(d_z))

doppz = np.array(doppz)

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
sns.heatmap(doppz[0:10,:,:].mean(axis=0))

def init():
    sns.heatmap(doppz[0:10,:,:].mean(axis=0))

def animate(i):
    sns.heatmap(doppz[i:i+10,:,:].mean(axis=0))

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, repeat=False)

plt.show()