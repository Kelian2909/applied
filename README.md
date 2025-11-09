

> Prédiction de la réadmission hospitalière de patients diabétiques à partir de données cliniques (UCI Machine Learning Repository)

---

## Description du projet

Ce projet s’appuie sur le jeu de données **[Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)** publié par l’UCI Machine Learning Repository.

L’objectif est de **prédire la probabilité de réadmission d’un patient diabétique dans les 30 jours** suivant une hospitalisation, à partir d’informations démographiques, médicales et administratives.

---

## Objectifs

- Comprendre les **facteurs associés à la réadmission** hospitalière des patients diabétiques.  
- .....
- .....


---

##  Étapes du projet

## 1. Nettoyage et prétraitement

- Remplacement des valeurs manquantes (`"?"`) par `NaN` puis suppression ou imputation.
- Suppression des colonnes inutilisables ou constantes (`weight`, `payer_code`, `examide`, etc.).
- Encodage ordinal des variables de médicaments (`"No" → 0`, `"Steady" → 1`, etc.).
- Définition des types de variables :
  - `num_cols` → variables numériques à standardiser (`StandardScaler`)
  - `ohe_cols` → variables catégorielles à encoder (`OneHotEncoder`)
  - `other_cols` → variables déjà numériques
- Construction d’un pipeline (`ColumnTransformer`) combinant toutes les transformations.
- Création de la variable cible binaire : `y = 1 si readmitted == "<30", sinon 0`.

---

## 2. Sélection de variables (à venir)

### 2.1. Information mutuelle

L’**information mutuelle** mesure la **dépendance statistique** entre une variable explicative `X` et la cible `Y`.  
Elle indique **combien d’information sur `Y` est contenue dans `X`**.  
Si `X` et `Y` sont indépendantes, elle est nulle.

C’est un **cas particulier généralisé du coefficient de corrélation** :  
- la corrélation de Pearson ne capture que les **relations linéaires**,  
- l’information mutuelle détecte **toutes les dépendances**, linéaires ou non.

#### Formule
\[
I(X;Y) = \sum_{x,y} p(x,y) \, \log \frac{p(x,y)}{p(x)p(y)}
\]
- `p(x, y)` : probabilité jointe  
- `p(x)`, `p(y)` : probabilités marginales  
- `I(X;Y) ≥ 0` et `I(X;Y) = 0` si indépendance

#### Interprétation
| Cas | Interprétation |
|------|----------------|
| `I(X;Y)=0` | X et Y indépendantes |
| `I(X;Y)>0` | X apporte de l’information sur Y |
| Score élevé | Variable pertinente |




- Méthodes filtre : variance, corrélation, `SelectKBest`.
- Méthodes embedded : `RandomForest`, `LogisticRegression (L1)`.
- Méthodes wrapper : `RFE`, `Boruta`, `SequentialFeatureSelector`.

L’objectif est de réduire la dimension, améliorer la performance et renforcer l’interprétabilité du modèle.

