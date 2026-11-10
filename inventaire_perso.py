# projet-machine_learning
#Séance: 1
#Nom : HOUNGBO Gwenael
#Porteur: P3
#Etapes Assignées: ##1.6-1.4-1.9-1.12

import pandas as pd   
from pathlib import Path  


//1.9
DATA = Path ("data")
train = pd.read_csv (DATA / "train.csv")
test = pd.read_csv (DATA / "a_estimer.csv")
print(train.shape, test.shape)

cat_cols  = [c for c in train.columns if c.startswith("cat")]
cont_cols = [c for c in train.columns if c.startswith("cont")]
print(len(cat_cols), len(cont_cols))



//1.6
print("---Trous ---")
print("NaN train :", train.isna().sum().sum())
print("NaN test  :", test.isna().sum().sum())

masques = ["", " ", "?", "NA", "N/A", "nan", "null", "None", "-"]
for c in cat_cols:
    n = train[c].isin(masques).sum()
    if n > 0:
        print(c, n)
print("contrôle des masques terminé")
for c in cont_cols:
    part = (train[c] == 0).mean()
    if part > 0.01:
        print(c, round(part, 3))
print("contrôle des zéros terminé")




// 1.4
print("--- Cardinalité ---")
card = train[cat_cols].nunique().sort_values(ascending=False)
print("min :", card.min(), "| max :", card.max())
print(card.head(10))
print("one-hot complet :", card.sum(), "colonnes")
print("variables > 255 modalités :", (card > 255).sum())