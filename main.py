import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier as knn
from textwrap import wrap

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
    'Obnizenie lizby RBC',
    'Liczba plytek krwi',
    'Niedojrzale komorki',
    'Stan pobudzenia szpiku',
    'Glowne komorki szpiku',
    'Poziom limfocytow',
    'Reakcja'
]

dfs = pd.read_excel("białaczka.xlsx")

chi_value, p_value = chi2(dfs[dfs.columns[:20]], dfs["Klasa"])

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

plt.show()