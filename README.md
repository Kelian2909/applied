

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

## 2. Sélection de variables

### 2.1. Détermination du nombre optimal de variables par validation croisée

Avant de procéder à la sélection proprement dite, il est essentiel de déterminer **combien de variables** il est pertinent de conserver.  
Pour cela, une **validation croisée (cross-validation)** a été réalisée à l’aide d’un modèle de **régression logistique**.

Le principe consiste à :
- Sélectionner un nombre de variables `k` selon un critère donné (ex. Information Mutuelle),
- Évaluer la performance du modèle (AUC) via une **Stratified K-Fold Cross-Validation**,
- Répéter l’opération pour différentes valeurs de `k`,
- Retenir le **nombre minimal de variables** donnant la **meilleure performance moyenne**.

Cette étape permet d’éviter la **sur-sélection** (trop de variables inutiles) tout en maximisant la **capacité prédictive** du modèle final.  
C’est une approche empirique, mais robuste, qui garantit un bon compromis entre **complexité** et **performance**.

---

### 2.2. Méthode 1 — Information Mutuelle (IM)

L’**information mutuelle** mesure la **dépendance statistique** entre une variable explicative `X` et la cible `Y`.  
Elle indique combien d’information sur `Y` est contenue dans `X`.  
Si `X` et `Y` sont indépendantes, elle est nulle.

Contrairement à la corrélation linéaire (Pearson), l’information mutuelle capte **toutes les formes de dépendance** — linéaires ou non linéaires.

\[
I(X;Y) = \sum_{x,y} p(x,y) \, \log \frac{p(x,y)}{p(x)p(y)}
\]

- `p(x, y)` : probabilité jointe  
- `p(x)` et `p(y)` : probabilités marginales  
- `I(X;Y) ≥ 0`, et `I(X;Y) = 0` si indépendance totale

L’IM est une **méthode de type filtre** :  
chaque variable est évaluée indépendamment du modèle, ce qui la rend rapide et générique.  
Elle permet d’identifier les variables **informativement pertinentes**, sans a priori sur la nature de la relation.

---

### 2.3. Méthode 2 — Régression Lasso (L1)

La **régression logistique pénalisée L1** (ou **Lasso**) fait partie des **méthodes embedded** :  
la sélection de variables est effectuée **pendant** l’entraînement du modèle.

Le principe repose sur l’ajout d’une **pénalisation absolue** sur les coefficients du modèle :

\[
\text{min} \; \|y - X\beta\|^2 + \lambda \sum_i |\beta_i|
\]

Sous l’effet du paramètre de régularisation `λ`, certains coefficients `βᵢ` deviennent **exactement nuls**, ce qui équivaut à **éliminer la variable correspondante**.  

Avantages :
- Sélection automatique des variables les plus explicatives,  
- Réduction du sur-apprentissage,  
- Maintien d’une bonne interprétabilité du modèle linéaire.

Le Lasso est particulièrement adapté lorsque plusieurs variables sont corrélées : il conserve celles qui apportent le plus d’information unique sur la cible.

---

### 2.4. Méthode 3 — XGBoost (Feature Importance)

Le modèle **XGBoost (Extreme Gradient Boosting)** est une méthode **arborescente** et **itérative** fondée sur le gradient boosting.  
Il combine de nombreux arbres de décision faibles pour construire un modèle global puissant.

Chaque variable se voit attribuer une **importance** mesurée par :
- la **fréquence d’utilisation** de la variable dans les arbres,
- la **réduction du gain d’erreur (Gain)** qu’elle procure lors d’une division,
- ou la **couverture (Cover)** des échantillons concernés.

Les importances sont ensuite normalisées pour obtenir un **score global** entre 0 et 1.  

Avantages :
- Capte les **interactions non linéaires** et **les effets croisés** entre variables,  
- Très robuste aux données bruitées ou corrélées,  
- Excellente performance empirique.

---

### 2.5. Pondération et agrégation des scores

Chaque méthode fournit une **mesure complémentaire de la pertinence** des variables :
- l’Information Mutuelle capture les dépendances statistiques,  
- le Lasso privilégie les variables linéairement discriminantes,  
- XGBoost identifie les contributions non linéaires dans un cadre de type arbre.

Pour obtenir une vision plus équilibrée, les trois scores ont été **normalisés entre 0 et 1** puis combinés en un **score global pondéré** :

\[
\text{Score}_{global} = 0.4 \times \text{IM} + 0.3 \times \text{Lasso} + 0.3 \times \text{XGBoost}
\]

Les poids ont été choisis pour donner une **légère priorité à la robustesse statistique (IM)**, tout en intégrant la **sélection structurelle (Lasso)** et la **non-linéarité (XGBoost)**.

Les variables présentant le plus haut score global ont été retenues pour constituer la base finale du modèle prédictif.

---

### 2.6. Bénéfices de l’approche combinée

Cette approche en trois étapes permet de :
- **Réduire la dimensionnalité** du jeu de données,  
- **Limiter la multicolinéarité**,  
- **Renforcer la stabilité** des variables retenues,  
- Et **combiner la rigueur statistique** (IM, Lasso) à la **puissance prédictive** des modèles non linéaires (XGBoost).

Ainsi, la sélection finale intègre à la fois :
- les dépendances statistiques pures,  
- les relations linéaires directes,  
- et les effets non linéaires complexes,  

assurant une description complète et robuste des variables les plus déterminantes.

---

### 2.7. Visualisation et interprétation

Les variables les mieux classées selon le score global sont visualisées sous forme d’un **bar chart horizontal**.  
La couleur orange met en évidence les **25 variables les plus pertinentes** retenues pour l’apprentissage final.  
Cette représentation permet d’évaluer d’un coup d’œil la contribution relative de chaque variable et d’identifier les dimensions les plus influentes du modèle.

---


