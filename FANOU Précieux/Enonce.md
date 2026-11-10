# Projet — Estimer le montant d'un sinistre

**Équipe de 3 · 6 semaines · rendu par dépôt**

---

## Le contexte

Un assureur reçoit la déclaration d'un sinistre. Il doit **estimer le montant** qu'il devra payer :
c'est ce montant qui alimente la provision du dossier et le tarif du risque.

L'estimation se fait à partir du dossier seul — les 130 variables qui le décrivent. Le montant, lui,
n'est connu qu'après règlement.

## La question à laquelle le projet répond

> Peut-on estimer le montant d'un sinistre à partir de son dossier — et de combien une méthode
> soignée bat-elle l'estimation que l'assureur applique aujourd'hui sans modèle ?

## Ce que vous recevez

| Fichier | Contenu |
|---|---|
| `train.csv` | **150 654 sinistres** dont le montant est connu — 132 colonnes, dont `loss` |
| `a_estimer.csv` | **37 664 sinistres** — les 131 mêmes colonnes, **sans `loss`** |

Les variables sont de deux sortes : **116 catégorielles** (`cat1`…`cat116`), de 2 à 326 modalités, et
**14 continues** (`cont1`…`cont14`). Toutes les colonnes sont anonymisées : il n'y a pas de sens
métier à retrouver, seulement des régularités à mesurer.

**Il n'y a aucune valeur manquante dans ce fichier.** C'est rare, et cela ne vous dispense de rien.

Aucune donnée ne permet d'identifier une personne : une ligne est un sinistre, et l'identifiant `id`
est un numéro interne.


## Ce que contient chaque colonne

| Famille | Nombre | Type | Ce que c'est |
|---|---|---|---|
| `id` | 1 | entier | identifiant du sinistre |
| `cat1` … `cat116` | **116** | **texte** (lettres `A`, `B`, `C`…) | variables catégorielles **anonymisées**, de **2 à 326 modalités** |
| `cont1` … `cont14` | **14** | décimal | variables continues **anonymisées** |
| `loss` | 1 | décimal | **la cible** : montant du sinistre, de 0,67 à 121 012 |

**Ce que l'on sait, et ce que l'on ne sait pas.** Aucune de ces variables n'a de signification publiée :
`cat57` n'est pas « le type de véhicule », c'est `cat57`. Vous ne pouvez donc pas raisonner par la
connaissance du métier — **vous devez mesurer ce que chaque variable apporte**, et c'est l'exercice.

**Ce qui est mesuré sur les données livrées :** aucune valeur manquante ; les 116 catégorielles ont de
2 à 326 modalités ; les 14 continues s'étendent de 0 à 587 602.


**Ce que Kaggle documente, et ce qu'il ne documente pas.** La page *Data* de cette compétition ne
publie **aucune description par colonne** : les variables sont anonymisées et la compétition ne donne
que leurs familles. **Les lignes ci-dessus sont donc tout ce qui est documenté** — vous ne trouverez
pas, en ligne, à quoi correspond une variable précise. C'est une contrainte du sujet, et elle est
volontaire : votre travail est de **mesurer** ce que chaque variable apporte, pas de l'interpréter.

## Ce que vous rendez

**Un dépôt de code**, pas un notebook :

1. un **pipeline exécutable** qui prend `a_estimer.csv` et produit un fichier `id,loss` ;
2. le **protocole de validation** retenu, justifié ;
3. les **essais**, y compris ceux qui ont échoué, avec leur score ;
4. une **note de 3 pages** : décision, métrique, méthode, résultat chiffré, ce qui a échoué ;
5. une **model card** ;
6. la **fiche de données** du S13.

Le pipeline doit tourner **en moins de 10 minutes sur CPU**, sans GPU, avec `scikit-learn` seul.

## Ce que l'enseignant garde — et ce que vous pouvez demander

**Vous recevez toutes les lignes, sauf mention contraire ci-dessous.** Ce qui reste chez l'enseignant
n'est **jamais** une partie des données : c'est **la cible** des lignes que vous devez estimer, et elle
seule. Sans elle, il n'y aurait rien à prédire.

| | |
|---|---|
| Fichier retenu | **`verite.csv`** |
| Lignes | **37 664** |
| Colonnes | `id`, `loss` |
| Contenu | le montant réel des sinistres de `a_estimer.csv` |

**Vous n'y avez pas accès, et aucune copie ne doit circuler.** C'est l'enseignant qui mesure votre
rendu contre ce fichier.

### La table source complète est disponible sur demande

Les données de ce sujet viennent d'une **compétition publique**. Si vous voulez travailler sur la
**table d'origine entière** — plus de lignes, plus de variables, ou les fichiers annexes — **demandez-la
à l'enseignant** : il l'a, et il peut vous la fournir pour un usage d'étude. Précisez dans votre
journal ce que vous en avez fait, et **restez sur la même découpe** : le rendu final porte sur les
lignes de `a_estimer.csv` / `a_decider.csv` / `horizon.csv`, pas sur la table complète.

## Travailler : récupérer les données, coder, sauvegarder

Les données du cours sont sur le **Drive du cours**, en lecture seule — **dossier du sujet :**
**https://drive.google.com/drive/folders/1v27scFmmiwfAH_QABh3cki8FX5ueoNL6?usp=sharing**. Vous téléchargez **le dossier de votre sujet** — pas les autres — et vous choisissez
ensuite **où coder**. Le lien est rappelé au **premier écran du carnet de démarrage**.

Le dossier contient **trois choses** : cet énoncé, le **carnet de démarrage** (`Carnet_de_demarrage.ipynb`)
et les fichiers de données.

### 1. Récupérer votre dossier

| | |
|---|---|
| Dossier du sujet | **`Sujet_04_Allstate/`** |
| Poids attendu | **70,3 Mo** |
| Contenu | les fichiers listés dans « Ce que vous recevez » |

Après téléchargement, vérifiez que rien n'est tronqué : le poids du dossier doit correspondre, et
chaque `csv` doit s'ouvrir avec le bon nombre de lignes.

### 2. Option A — travailler en local, dans VS Code

C'est la voie la plus simple : vos fichiers sont sur votre disque, rien ne disparaît.

```
python -m venv .venv
```

```
source .venv/bin/activate
```

*(sous Windows : `.venv\Scripts\activate`)*

```
pip install pandas scikit-learn
```

Ouvrez le dossier dans VS Code, écrivez votre pipeline — `pipeline.py` par exemple — et lancez-le :

```
python pipeline.py
```

**La contrainte des 10 minutes s'applique à votre machine.** Mesurez-la : elle fait partie du
livrable, et un pipeline qui tourne en 40 minutes sur votre portable ne sera pas accepté.

### 3. Option B — travailler dans un carnet Kaggle

Kaggle ne voit pas votre disque : il faut **lui donner vos fichiers**, en créant votre propre jeu de
données. C'est vous qui le déposez, pour votre usage — le jeu est **privé par défaut**.

1. **kaggle.com/datasets** → **New Dataset**.
2. Déposez les fichiers du dossier `Sujet_04_Allstate/` (glisser-déposer), nommez le jeu de données
   `esgc-vak-sujet-04-…`, puis **Create**.
3. Dans votre carnet : **Add Input** → onglet **Datasets** → filtre **Your Datasets** → sélectionnez
   le vôtre → **Add**.
4. Les fichiers apparaissent dans **`/kaggle/input/<nom-de-votre-jeu>/`**, en **lecture seule**.
   Vérifiez avant d'écrire du code :

```
!ls /kaggle/input/<nom-de-votre-jeu>/
```

5. Réglez la session : **Session options** → **Accelerator : None** — le projet tient sur CPU, c'est
   la contrainte du cours. Laissez **Internet désactivé**.

> **La compétition d'origine n'est pas la source.** Ne cherchez pas à l'ajouter : ses fichiers
> diffèrent des vôtres — pas d'horizon sans cible, pas de jeu à estimer — et plusieurs de ces
> compétitions ne sont plus rejoignables.

### 4. Sauvegarder, et réutiliser votre travail

**En local (Option A).** Votre dossier de travail **est** la sauvegarde. Gardez les sorties dans un
sous-dossier `sorties/`, et versionnez avec `git` si vous le souhaitez.

**Sur Kaggle (Option B).** `/kaggle/input` est en **lecture seule** : écrivez dans
**`/kaggle/working/`**.

```python
import os
os.makedirs("/kaggle/working/sorties", exist_ok=True)
prediction.to_csv("/kaggle/working/sorties/prediction.csv", index=False)
```

- **Sauvegarder** : bouton **Save Version** → **Save & Run All (Commit)**. Le carnet est **ré-exécuté
  de bout en bout** dans un conteneur propre, et **tout ce qui reste dans `/kaggle/working/` est
  conservé** comme sortie de cette version.
- **Reprendre ces sorties dans un autre carnet** : **Add Input** → onglet **Notebook Output** →
  votre carnet, la version voulue → le contenu apparaît dans `/kaggle/input/<nom-du-carnet>/`.
- **⚠️ `/kaggle/working` est éphémère** : une session inactive est coupée au bout de 20 minutes. Sans
  **Save Version**, tout est perdu.

**Quel que soit votre choix**, le livrable est le même : un **dépôt de code** reproductible, dont le
pipeline tourne en moins de 10 minutes sur CPU.

## Les trois temps du projet

### Bloc 1 — avant la séance 5 · *la décision et la mesure*

- **La métrique** : laquelle vous adoptez, et **ce qu'elle implique**. Une métrique ne dit pas
  seulement comment vous serez jugés : elle détermine ce qu'est une bonne réponse.
- **La constante de référence** : si l'on n'a aucun modèle, quelle valeur unique faut-il annoncer ?
  La réponse dépend de la métrique, et elle doit être **mesurée**, pas supposée.
- **Le coût** : que coûte une sous-estimation, que coûte une surestimation ? Sont-elles symétriques ?
- **Le protocole de validation** : comment vous découpez `train.csv` pour estimer honnêtement votre
  performance sur `a_estimer.csv` — et pourquoi ce découpage-là. La cible est un **montant**, pas une
  classe : cela change la façon de découper.

### Bloc 2 — avant la séance 9 · *le pipeline*

- Le pipeline complet, exécutable d'un bout à l'autre.
- **Au moins huit essais**, dont ceux qui ont échoué, chacun avec son score.
- **L'encodage des 116 variables catégorielles** : ce que vous faites, et pourquoi. Certaines ont
  plus de 300 modalités ; l'encodage le plus évident n'est pas toujours tenable.
- L'**analyse d'erreurs** : sur quels sinistres votre modèle se trompe, et de combien.

### Bloc 3 — avant la séance 13 · *le dépôt et la robustesse*

- Le dépôt propre, la note de 3 pages, la model card, la fiche de données.
- **Trois mutations nommées**, chacune applicable en **une cellule**.
- Le test de robustesse, et l'estimation finale des 37 664 sinistres.

## Comment vous êtes évalués

Voir la grille du cours. Trois points pèsent particulièrement ici :

- **la constante de référence** : votre score se rapporte à une estimation sans modèle, mesurée, pas
  à zéro ;
- **la cohérence entre la métrique et la décision** : un modèle excellent mesuré par la mauvaise
  métrique est un mauvais modèle ;
- **la vérité du journal** : le journal consigne les essais, y compris ceux qui ont échoué.

## Deux règles du cours

**Pas de fuite.** Une information qui ne serait pas connue **au moment où le dossier est instruit** ne
peut pas entrer dans le modèle.

**Le code doit rester réparable.** On doit pouvoir passer d'une version à l'autre en changeant une
cellule, pas en réécrivant le pipeline.
