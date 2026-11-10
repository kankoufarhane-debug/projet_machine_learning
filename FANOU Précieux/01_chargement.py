from IPython.display import display

# ── Ce que fait ce bloc : importer les outils et définir où sont les fichiers ──
from pathlib import Path      # pour manipuler des chemins de fichiers proprement
import pandas as pd           # pour lire les CSV et manipuler les tableaux (DataFrame)

# Le notebook est dans le même dossier que les CSV : on utilise des chemins relatifs
CHEMIN_TRAIN = Path("train.csv")          # sinistres AVEC le montant (loss)
CHEMIN_A_ESTIMER = Path("a_estimer.csv")  # sinistres SANS le montant

# On vérifie que les fichiers existent avant de les lire
print("train.csv existe     :", CHEMIN_TRAIN.exists())
print("a_estimer.csv existe :", CHEMIN_A_ESTIMER.exists())

# ── Résultat attendu ──
# train.csv existe     : True
# a_estimer.csv existe : True
# (si False : le notebook n'est pas dans le dossier Sujet_04_Allstate)

# ── Ce que fait ce bloc : lire train.csv et afficher sa forme ──
train = pd.read_csv(CHEMIN_TRAIN)   # lecture simple, sans typage particulier (étape 1.3)

print("Forme de train :", train.shape)                 # (nombre de lignes, nombre de colonnes)
print("Lignes         :", train.shape[0])
print("Colonnes       :", train.shape[1])

# ── Résultat attendu ──
# Forme de train : (150654, 132)
# Lignes         : 150654
# Colonnes       : 132

# ── Ce que fait ce bloc : lire a_estimer.csv et afficher sa forme ──
a_estimer = pd.read_csv(CHEMIN_A_ESTIMER)

print("Forme de a_estimer :", a_estimer.shape)
print("Lignes             :", a_estimer.shape[0])
print("Colonnes           :", a_estimer.shape[1])

# ── Résultat attendu ──
# Forme de a_estimer : (37664, 131)
# Lignes             : 37664
# Colonnes           : 131   (une de moins que train : pas de colonne loss)

# ── Ce que fait ce bloc : vérifier que a_estimer a les mêmes colonnes que train, sauf loss ──
colonnes_train = list(train.columns)
colonnes_a_estimer = list(a_estimer.columns)

# Colonnes présentes dans train mais absentes de a_estimer (on attend seulement 'loss')
seulement_train = [c for c in colonnes_train if c not in colonnes_a_estimer]
# Colonnes présentes dans a_estimer mais absentes de train (on attend aucune)
seulement_a_estimer = [c for c in colonnes_a_estimer if c not in colonnes_train]

print("Seulement dans train     :", seulement_train)
print("Seulement dans a_estimer :", seulement_a_estimer)

# Même ordre des colonnes ? (utile pour appliquer plus tard le même pipeline aux deux)
meme_ordre = colonnes_a_estimer == [c for c in colonnes_train if c != "loss"]
print("Même ordre des colonnes (hors loss) :", meme_ordre)

# ── Résultat attendu ──
# Seulement dans train     : ['loss']
# Seulement dans a_estimer : []
# Même ordre des colonnes (hors loss) : True

# ── Ce que fait ce bloc : regarder à quoi ressemble le tableau ──
# On affiche seulement quelques colonnes pour que ce soit lisible :
# id, les 3 premières catégorielles, les 3 premières continues, et la cible loss
colonnes_apercu = ["id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3", "loss"]
print(train[colonnes_apercu].head())

# ── Résultat attendu ──
# Un tableau de 5 lignes. cat1, cat2, cat3 contiennent des lettres (A, B...),
# cont1, cont2, cont3 et loss des décimaux, id un entier.

# ── Ce que fait ce bloc : regrouper tous les contrôles de l'étape dans un seul tableau ──
nb_communes = len(set(train.columns) & set(a_estimer.columns))   # colonnes présentes dans les deux

# Chaque contrôle : (nom, valeur attendue d'après l'énoncé, valeur obtenue)
controles = [
    ("Lignes de train",       150654, train.shape[0]),
    ("Colonnes de train",     132,    train.shape[1]),
    ("Lignes de a_estimer",   37664,  a_estimer.shape[0]),
    ("Colonnes de a_estimer", 131,    a_estimer.shape[1]),
    ("Colonnes communes",     131,    nb_communes),
]

recap = pd.DataFrame(controles, columns=["Contrôle", "Attendu", "Obtenu"])
# Verdict : OK si obtenu == attendu, sinon ATTENTION
recap["Verdict"] = recap.apply(
    lambda ligne: "OK" if ligne["Obtenu"] == ligne["Attendu"] else "ATTENTION", axis=1
)
display(recap)

# Message final clair
if (recap["Verdict"] == "OK").all():
    print("Étape 1.2 validée : toutes les formes correspondent à l'énoncé.")
else:
    print("ATTENTION : au moins une forme ne correspond pas. Fichier tronqué ou mauvais fichier ?")

# ── Résultat attendu ──
# Un tableau de 5 lignes, toutes en OK, puis :
# Étape 1.2 validée : toutes les formes correspondent à l'énoncé.