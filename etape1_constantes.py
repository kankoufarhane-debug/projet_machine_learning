import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("train.csv")
y = df["loss"].to_numpy()

# 1. Les trois constantes candidates
print("moyenne          :", y.mean())
print("médiane          :", np.median(y))
print("moyenne géom.    :", np.exp(np.log(y).mean()))
print("asymétrie (skew) :", pd.Series(y).skew())

# 2. Score de chaque constante en validation croisée
def mae(y, c):
    return np.mean(np.abs(y - c))

def rmse(y, c):
    return np.sqrt(np.mean((y - c) ** 2))

consts = {
    "moyenne": np.mean,
    "médiane": np.median,
    "moy. géom.": lambda a: np.exp(np.log(a).mean()),
}
res = {k: {"MAE": [], "RMSE": []} for k in consts}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr, va in kf.split(y):
    for nom, f in consts.items():
        c = f(y[tr])
        res[nom]["MAE"].append(mae(y[va], c))
        res[nom]["RMSE"].append(rmse(y[va], c))

print()
for nom, r in res.items():
    print(f"{nom:11s} MAE={np.mean(r['MAE']):9.1f}  RMSE={np.mean(r['RMSE']):9.1f}")