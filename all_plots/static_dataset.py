import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
labels = ['looking\nforward', 'Talking', 'yawning', 'looking\nright', 'looking\nleft']

array = [[289, 37, 43, 8, 9],
         [56, 100, 17, 7, 2],
         [16, 1, 156, 0, 1],
         [8, 4, 2, 152, 0],
         [10, 0, 0, 0, 162]]

array = np.array(array)
array = array / array.sum(axis=0)
df_cm = pd.DataFrame(array, index=[i for i in labels], columns=[i for i in labels])
# plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.yticks(rotation=25)
plt.xticks(rotation=25)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
