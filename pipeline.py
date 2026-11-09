"""
Projet Allstate - estimer le montant d'un sinistre.
Rôle : constructeur (chargement, variables, encodage, pipeline reproductible).

Usage :
    python pipeline.py                                  # essai avec la config par défaut
    python pipeline.py --encodage frequence             # mutation 1 : l'encodage
    python pipeline.py --cible-log                      # mutation 2 : cible en log
    python pipeline.py --modele ridge --encodage onehot # mutation 3 : le modèle
    python pipeline.py --nom essai_04 --sans-final      # juste la validation, sans prédire

Chaque exécution ajoute une ligne à sorties/essais.csv (essais réussis ET ratés).
Le fichier sorties/prediction.csv (id,loss) est écrit à la fin.
Contrainte : CPU seul, scikit-learn seul, < 10 minutes.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

# ----------------------------------------------------------------------------
# CONFIGURATION : les trois mutations se font ici (ou en ligne de commande)
# ----------------------------------------------------------------------------
CONFIG = {
    "data_dir": "data",          # dossier contenant train.csv et a_estimer.csv
    "sorties": "sorties",
    "encodage": "ordinal",       # "ordinal" | "natif" | "frequence" | "target" | "onehot"
    "cible_log": False,          # True : on apprend log1p(loss) puis on revient en euros
    "modele": "hgb",             # "hgb" | "ridge"
    "perte": "absolute_error",   # perte de HGB : "absolute_error" (MAE) | "squared_error"
    "n_splits": 5,               # plis de validation
    "n_bins_strat": 10,          # tranches de montant pour stratifier le découpage
    "seed": 42,
    "max_iter": 300,             # nombre d'arbres de HGB
}


# ----------------------------------------------------------------------------
# 1. CHARGEMENT
# ----------------------------------------------------------------------------
def charger(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "a_estimer.csv")

    # Vérifications annoncées par l'énoncé : si ça casse, le fichier est tronqué.
    assert train.shape == (150_654, 132), f"train inattendu : {train.shape}"
    assert test.shape == (37_664, 131), f"a_estimer inattendu : {test.shape}"
    assert "loss" not in test.columns
    assert train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0

    cat_cols = [c for c in train.columns if c.startswith("cat")]
    cont_cols = [c for c in train.columns if c.startswith("cont")]
    assert len(cat_cols) == 116 and len(cont_cols) == 14

    # float32 pour les continues : mémoire et vitesse
    train[cont_cols] = train[cont_cols].astype("float32")
    test[cont_cols] = test[cont_cols].astype("float32")
    return train, test, cat_cols, cont_cols


# ----------------------------------------------------------------------------
# 2. ENCODAGE DES 116 CATÉGORIELLES
# ----------------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Modalité -> sa fréquence dans le train. Modalité inconnue -> 0. Pas de fuite de cible."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(str)
        self.maps_ = [X.iloc[:, j].value_counts(normalize=True) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str)
        out = np.empty(X.shape, dtype="float32")
        for j, m in enumerate(self.maps_):
            out[:, j] = X.iloc[:, j].map(m).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def construire_preprocesseur(encodage, cat_cols, cont_cols, seed):
    """Retourne (preprocesseur, categorical_features_pour_hgb)."""
    cats_hgb = None  # masque des colonnes catégorielles natives de HGB (si utilisé)

    if encodage == "ordinal":
        # Entiers arbitraires : les arbres s'en accommodent. Inconnu -> NaN (géré par HGB).
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, dtype="float32")
    elif encodage == "natif":
        # HGB gère les catégories nativement, mais seulement jusqu'à 255 modalités par
        # variable : on regroupe les plus rares (cat jusqu'à 326 modalités).
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan,
            max_categories=254, dtype="float32",
        )
        cats_hgb = np.array([True] * len(cat_cols) + [False] * len(cont_cols))
    elif encodage == "frequence":
        enc = FrequencyEncoder()
    elif encodage == "target":
        # TargetEncoder ajuste avec une validation croisée interne : pas de fuite
        # dans fit_transform (le piège principal de cet encodage).
        enc = TargetEncoder(target_type="continuous", cv=5, random_state=seed)
    elif encodage == "onehot":
        # Pour les modèles linéaires. Modalités rares regroupées pour limiter les colonnes.
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005)
    else:
        raise ValueError(f"encodage inconnu : {encodage}")

    cont_step = StandardScaler() if encodage == "onehot" else "passthrough"
    pre = ColumnTransformer(
        [("cat", enc, cat_cols), ("cont", cont_step, cont_cols)],
        sparse_threshold=0.3,
    )
    return pre, cats_hgb


def construire_modele(cfg, cat_cols, cont_cols):
    pre, cats_hgb = construire_preprocesseur(cfg["encodage"], cat_cols, cont_cols, cfg["seed"])
    if cfg["modele"] == "hgb":
        reg = HistGradientBoostingRegressor(
            loss=cfg["perte"], max_iter=cfg["max_iter"], learning_rate=0.1,
            categorical_features=cats_hgb, random_state=cfg["seed"],
        )
    elif cfg["modele"] == "ridge":
        reg = Ridge(alpha=10.0)
    else:
        raise ValueError(f"modèle inconnu : {cfg['modele']}")

    pipe = Pipeline([("pre", pre), ("reg", reg)])
    if cfg["cible_log"]:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


# ----------------------------------------------------------------------------
# 3. VALIDATION (la cible est un montant : on stratifie sur des tranches de montant)
# ----------------------------------------------------------------------------
def decoupages(y, cfg):
    tranches = pd.qcut(y, q=cfg["n_bins_strat"], labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    return list(skf.split(np.zeros(len(y)), tranches))


def valider(cfg, X, y, cat_cols, cont_cols):
    """MAE en validation croisée + prédictions hors-pli (pour l'analyse d'erreurs)."""
    oof = np.zeros(len(y))
    maes = []
    for k, (itr, iva) in enumerate(decoupages(y, cfg)):
        m = construire_modele(cfg, cat_cols, cont_cols)
        m.fit(X.iloc[itr], y.iloc[itr])
        oof[iva] = m.predict(X.iloc[iva])
        maes.append(mean_absolute_error(y.iloc[iva], oof[iva]))
        print(f"  pli {k + 1}/{cfg['n_splits']} : MAE = {maes[-1]:,.2f}")
    return float(np.mean(maes)), float(np.std(maes)), oof


def erreur_par_tranche(y, oof):
    """Où le modèle se trompe : MAE sur l'ensemble, les 90 % plus légers, les 10 % et 1 % plus lourds."""
    y = np.asarray(y)
    q90, q99 = np.quantile(y, [0.90, 0.99])
    err = np.abs(y - oof)
    return {
        "mae_tout": err.mean(),
        "mae_90_legers": err[y <= q90].mean(),
        "mae_10_lourds": err[y > q90].mean(),
        "mae_1_lourds": err[y > q99].mean(),
        "biais_moyen": (oof - y).mean(),  # >0 : on surestime en moyenne
    }


def planchers(y):
    """Constantes de référence, mesurées en MAE sur le train (moyenne vs médiane)."""
    return {
        "plancher_moyenne": float(np.mean(np.abs(y - y.mean()))),
        "plancher_mediane": float(np.mean(np.abs(y - y.median()))),
    }


# ----------------------------------------------------------------------------
# 4. JOURNAL DES ESSAIS (on consigne aussi les échecs)
# ----------------------------------------------------------------------------
def journaliser(sorties, ligne):
    chemin = Path(sorties) / "essais.csv"
    df = pd.DataFrame([ligne])
    df.to_csv(chemin, mode="a", header=not chemin.exists(), index=False)


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main(cfg, nom, sans_final):
    t0 = time.time()
    Path(cfg["sorties"]).mkdir(parents=True, exist_ok=True)

    train, test, cat_cols, cont_cols = charger(cfg["data_dir"])
    features = cat_cols + cont_cols
    X, y = train[features], train["loss"]

    pl = planchers(y)
    print(f"Planchers (MAE) : moyenne = {pl['plancher_moyenne']:,.2f} | médiane = {pl['plancher_mediane']:,.2f}")
    print(f"Essai '{nom}' : {cfg['encodage']} / {cfg['modele']} / log={cfg['cible_log']} / perte={cfg['perte']}")

    ligne = {
        "date": datetime.now().isoformat(timespec="seconds"), "nom": nom,
        "encodage": cfg["encodage"], "modele": cfg["modele"],
        "cible_log": cfg["cible_log"], "perte": cfg["perte"],
        "mae_cv": np.nan, "ecart_type": np.nan, "temps_s": np.nan, "statut": "",
    }
    try:
        mae, std, oof = valider(cfg, X, y, cat_cols, cont_cols)
        ligne.update(mae_cv=round(mae, 2), ecart_type=round(std, 2), statut="ok")
        print(f"MAE validation croisée : {mae:,.2f} (± {std:,.2f})")
        for k, v in erreur_par_tranche(y, oof).items():
            print(f"  {k:>15} : {v:,.2f}")
    except Exception as e:  # un essai raté est un résultat : on le garde dans le journal
        ligne["statut"] = f"echec : {type(e).__name__}: {e}"[:200]
        ligne["temps_s"] = round(time.time() - t0, 1)
        journaliser(cfg["sorties"], ligne)
        raise

    if not sans_final:
        modele = construire_modele(cfg, cat_cols, cont_cols)
        modele.fit(X, y)
        pred = np.clip(modele.predict(test[features]), 0, None)  # un montant n'est jamais négatif
        pd.DataFrame({"id": test["id"], "loss": pred}).to_csv(
            Path(cfg["sorties"]) / "prediction.csv", index=False
        )
        print(f"Prédictions écrites : {Path(cfg['sorties']) / 'prediction.csv'}")

    ligne["temps_s"] = round(time.time() - t0, 1)
    journaliser(cfg["sorties"], ligne)
    print(f"Temps total : {ligne['temps_s']} s (limite : 600 s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nom", default="essai")
    p.add_argument("--data-dir")
    p.add_argument("--encodage", choices=["ordinal", "natif", "frequence", "target", "onehot"])
    p.add_argument("--modele", choices=["hgb", "ridge"])
    p.add_argument("--perte", choices=["absolute_error", "squared_error"])
    p.add_argument("--cible-log", action="store_true")
    p.add_argument("--n-splits", type=int)
    p.add_argument("--sans-final", action="store_true", help="validation seule, pas de prédiction finale")
    a = p.parse_args()

    cfg = dict(CONFIG)
    for cle in ["data_dir", "encodage", "modele", "perte", "n_splits"]:
        if getattr(a, cle) is not None:
            cfg[cle] = getattr(a, cle)
    if a.cible_log:
        cfg["cible_log"] = True
    main(cfg, a.nom, a.sans_final)
