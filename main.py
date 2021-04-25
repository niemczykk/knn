import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import cross_val_score
from statistics import mean
from textwrap import wrap

clfs = {
    'k1euclidean': knc(n_neighbors=1, metric='euclidean'),
    'k5euclidean': knc(n_neighbors=5, metric='euclidean'),
    'k10euclidean': knc(n_neighbors=10, metric='euclidean'),
    'k1manhattan': knc(n_neighbors=1, metric='manhattan'),
    'k5manhattan': knc(n_neighbors=5, metric='manhattan'),
    'k10manhattan': knc(n_neighbors=10, metric='manhattan')
}

cechy = [
    'Temperatura',
    'Anemia',
    'Stopien krwawienia',
    'Miejsce krwawienia',
    'Bol kosci',
    'Wrazliwosc mostka',
    'Powiekszenie wezlow chlonnych',
    'Powiekszenie watroby i sledziony',
    'Centralny uklad nerwowy',
    'Powiekszenie jader',
    'Uszkodzenie w sercu, plucach, nerce',
    'Galka oczna',
    'Poziom WBC',
    'Obnizenie lizby RRC',
    'Liczba plytek krwi',
    'Niedojrzale komorki',
    'Stan pobudzenia szpiku',
    'Glowne komorki szpiku',
    'Poziom limfocytow',
    'Reakcja'
]

dfs = pd.read_excel("białaczka.xlsx")
X = dfs[dfs.columns[:20]]
y = dfs["Klasa"]

chi_value, p_value = chi2(X, y)

unsorted = list(zip(cechy, chi_value))
unsorted_2 = sorted(unsorted, key=lambda x: x[1])
sorted_s = [[i for i, j in unsorted_2], [j for i, j in unsorted_2]]

y_pos = np.arange(len(chi_value))
labels = ['\n'.join(wrap(i, width=20)) for i in sorted_s[0]]

plt.figure(figsize=(15, 15))
plt.ylabel('Cecha')
plt.xlabel('Chi statistic')
plt.barh(y_pos, sorted_s[1])
plt.yticks(y_pos, labels)

for i in range(20):
    plt.text(sorted_s[1][i], i, '%.3f' % sorted_s[1][i])

dfs_list = np.array(dfs.values.tolist(), dtype='int64')
val = []
for key in clfs:
    print(key)
    used = [[] for i in range(410)]
    for i in range(len(sorted_s[0]) - 1, -1, -1):
        print(i)
        for index, tab in enumerate(used):
            tab.append(dfs_list[index, dfs.columns.tolist().index(sorted_s[0][i])])
        for j in range(4):
            scores = cross_val_score(clfs[key], used, y, cv=2, scoring='accuracy', n_jobs=4)
            val.append(scores.mean())
        print(mean(val))