import sys
from library.augumentation import Dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.append('../')


def PoolOp(img, ksize=(16, 16), op=np.mean):
    arr = []
    i_h, i_w, i_c = img.shape
    k_h, k_w = ksize
    row = []
    for c in range(i_c):
        for i in range(i_h // k_h):
            for j in range(i_w // k_w):
                row.append(op(img[k_h * i:k_h * i + k_h, k_w * j:k_w * j + k_w, c]))
    return np.array(row)


preprocess = lambda v: np.concatenate([PoolOp(v, op=np.min),
                                       PoolOp(v, op=np.max),
                                       PoolOp(v, op=np.mean),
                                       PoolOp(v, op=np.std)]).flatten().tolist()
data = Dataset(loc='../static_dataset/processed/denoised/*',
               class_count=1000,
               frame_stack=2,
               dop_min=1375,
               dop_max=5293, )

lbl_map = \
    {'looking forward': 0,
     'Talking': 1,
     'yawning': 2,
     'looking right': 3,
     'looking left': 4,
     'looking up': 5}

label = np.array(list(map(lambda e: lbl_map[e], data.label)))

X = np.array([preprocess(d) for d in data.data])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_train)

print(classification_report(y_train, pred))
pred = rf.predict(X_test)

print(classification_report(y_test, pred))

sns.heatmap(confusion_matrix(y_test, pred), annot=True)
plt.xticks(ticks=list(lbl_map.values()), labels=list(lbl_map.keys()), rotation=45)
plt.yticks(ticks=list(lbl_map.values()), labels=list(lbl_map.keys()), rotation=45)
plt.tight_layout()
