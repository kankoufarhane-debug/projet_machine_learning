import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

df = pd.read_csv("train.csv")
y = df["loss"].to_numpy()

# 1. Forme de la cible
print(pd.Series(y).describe(percentiles=[.01, .1, .25, .5, .75, .9, .99, .999]))
top1 = np.sort(y)[-len(y) // 100:].sum() / y.sum()
print(f"\nPart des 1 % plus gros sinistres dans le total : {top1:.1%}")

# 2. Découpage simple vs stratifié sur déciles de loss
bins = pd.qcut(y, q=10, labels=False)

def ecart_entre_plis(splitter_cls, graine):
    sp = splitter_cls(n_splits=5, shuffle=True, random_state=graine)
    maes = []
    for tr, va in sp.split(y, bins):
        c = np.median(y[tr])
        maes.append(np.mean(np.abs(y[va] - c)))
    return np.std(maes)

for nom, cls in [("KFold simple", KFold), ("Stratifié (déciles)", StratifiedKFold)]:
    ecarts = [ecart_entre_plis(cls, g) for g in range(20)]
    print(f"{nom:22s} écart-type de la MAE entre plis : {np.mean(ecarts):.2f}")