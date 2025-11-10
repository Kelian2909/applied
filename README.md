

> Prédiction de la réadmission hospitalière de patients diabétiques à partir de données cliniques (UCI Machine Learning Repository)



## Description du projet

Ce projet s’appuie sur le jeu de données **[Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)** publié par l’UCI Machine Learning Repository.

L’objectif est de **prédire la probabilité de réadmission d’un patient diabétique dans les 30 jours** suivant une hospitalisation, à partir d’informations démographiques, médicales et administratives.



## Objectifs

- Comprendre les **facteurs associés à la réadmission** hospitalière des patients diabétiques.  
- .....
- .....




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



## 2. Sélection de variables

### 2.1. Détermination du nombre optimal de variables par cross validation

Avant de procéder à la sélection, il est essentiel de déterminer **combien de variables** il est pertinent de conserver.  
Pour cela, une **cross-validation** a été réalisée à l’aide d’un modèle de **régression logistique**.

Le principe consiste à :
- Sélectionner un nombre de variables `k` selon un critère donné,
- Évaluer la performance du modèle (AUC) via une **Stratified K-Fold Cross-Validation**,
- Répéter l’opération pour différentes valeurs de `k`,
- Retenir le **nombre minimal de variables** donnant la **meilleure performance moyenne**.

Cette étape permet d’éviter la **sur-sélection** tout en maximisant la **capacité prédictive** du modèle final.  
C’est une approche empirique, mais robuste, qui garantit un bon compromis entre **complexité** et **performance**.



### 2.2. Méthode 1 — Information Mutuelle (IM)

L’**information mutuelle** mesure la **dépendance statistique** entre une variable explicative `X` et la cible `Y`.  
Elle indique combien d’information sur `Y` est contenue dans `X`.  
Si `X` et `Y` sont indépendantes, elle est nulle.

Contrairement à la corrélation linéaire (Pearson), l’information mutuelle capte **toutes les formes de dépendance** — linéaires ou non linéaires.

`I(X;Y) = Σₓ,ᵧ p(x,y) * log( p(x,y) / [p(x) * p(y)] )`

- `p(x, y)` : probabilité jointe  
- `p(x)` et `p(y)` : probabilités marginales  
- `I(X;Y) ≥ 0`, et `I(X;Y) = 0` si indépendance totale

L’IM est une **méthode de type filtre** :  
chaque variable est évaluée indépendamment du modèle, ce qui la rend rapide et générique.  
Elle permet d’identifier les variables **informativement pertinentes**, sans a priori sur la nature de la relation.


### 2.3. Méthode 2 — Régression Lasso (L1)

La **régression logistique pénalisée L1** (ou **Lasso**) fait partie des **méthodes embedded** :  
la sélection de variables est effectuée **pendant** l’entraînement du modèle.

Le principe repose sur l’ajout d’une **pénalisation absolue** sur les coefficients du modèle :

`minimize ||y - Xβ||² + λ Σ |βᵢ|`


Sous l’effet du paramètre de régularisation `λ`, certains coefficients `βᵢ` deviennent **exactement nuls**, ce qui équivaut à **éliminer la variable correspondante**.  

Avantages :
- Sélection automatique des variables les plus explicatives,  
- Réduction du sur-apprentissage,  
- Maintien d’une bonne interprétabilité du modèle linéaire.





### 2.4. Méthode 3 — XGBoost 

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


### 2.5. Pondération et agrégation des scores

Chaque méthode fournit une **mesure complémentaire de la pertinence** des variables :
- l’Information Mutuelle capture les dépendances statistiques,  
- le Lasso privilégie les variables linéairement discriminantes,  
- XGBoost identifie les contributions non linéaires dans un cadre de type arbre.

Pour obtenir une vision plus équilibrée, les trois scores ont été **normalisés entre 0 et 1** puis combinés en un **score global pondéré** :

`Score_global = 0.4 × IM + 0.3 × Lasso + 0.3 × XGBoost`


Les poids ont été choisis pour donner une **légère priorité à la robustesse statistique (IM)**, tout en intégrant la **sélection structurelle (Lasso)** et la **non-linéarité (XGBoost)**.

Les variables présentant le plus haut score global ont été retenues pour constituer la base finale du modèle prédictif.



## 3. Modélisation et évaluation

### 3.1. Méthodologie d’évaluation

La variable cible `readmitted` étant fortement **déséquilibrée** (~11 % de réadmissions), le critère principal choisi est la **PR-AUC (Precision–Recall Area Under Curve)**.  
Cette métrique évalue la capacité du modèle à **identifier les patients réadmis** (rappel) tout en **limitant les fausses alertes** (précision).  
Elle est plus adaptée qu’une ROC-AUC dans le cas d’un déséquilibre important entre classes.

Critères utilisés :
- **PR-AUC** : métrique principale.  
- **ROC-AUC** : performance globale de classement.  
- **F1-score** : compromis précision / rappel au seuil optimal.  
- **Brier Score** : mesure de calibration des probabilités.  
- **Recall@Top 20 %** : taux de vrais positifs dans les 20 % des patients les plus à risque.

Les données sont séparées en **80 % train / 20 % test** (stratifié).  
Tous les prétraitements (scaling, encodage, sélection de variables) sont inclus dans un **pipeline scikit-learn**, assurant l’absence de fuite de données.

---

### 3.2. Modèle interprétable — Régression Logistique L1 (Lasso)

Le modèle **Logistic Regression L1** a été choisi pour sa **transparence** et sa capacité à **sélectionner automatiquement les variables pertinentes**.  
Il constitue une première approche interprétable et robuste.

**Paramètres principaux :**
```python
LogisticRegression(
    penalty="l1",
    solver="liblinear",
    class_weight="balanced",
    max_iter=200,
    random_state=42
)
---
### 3.2. Validation croisée

**Méthodologie :**
- **5 folds** : `StratifiedKFold`
- **Scoring** : `{"pr_auc": "average_precision", "roc_auc": "roc_auc"}`

**Résultats (validation moyenne ± écart-type)** :

| Modèle      | PR-AUC (± std) | ROC-AUC (± std) |
|--------------|----------------|-----------------|
| LogReg L1 | 0.197 ± 0.005 | 0.638 ± 0.007 |
| LogReg L2 | 0.197 ± 0.005 | 0.638 ± 0.007 |

> ✅ **LogReg L1** retenue pour son caractère parcimonieux et interprétable.

---

### 3.3. Résultats sur jeu de test

| Métrique | Score |
|-----------|--------|
| **PR-AUC** | 0.193 |
| **ROC-AUC** | 0.633 |
| **F1-score (seuil = 0.49)** | 0.253 |
| **Recall (classe 1)** | 0.529 |
| **Precision (classe 1)** | 0.166 |
| **Brier Score** | 0.232 |

Le seuil a été déterminé en **maximisant le F1-score**.  
Le modèle identifie environ **53 % des patients réadmis**, au prix d’un taux modéré de faux positifs — un compromis acceptable en contexte médical.

---

### 3.4. Calibration

Le **Brier score (0.232)** montre une **calibration moyenne** :  
le modèle tend à **sous-estimer les risques** pour les patients à forte probabilité de réadmission.  
La **courbe de calibration** reste globalement cohérente avec la diagonale idéale.

<p align="center">
  <img src="outputs/calibration_curve_logreg_l1.png" width="480">
</p>

---

### 3.5. Interprétation du modèle

Les coefficients de la **régression logistique L1** permettent une lecture directe de l’influence de chaque variable :

- **β > 0** → la variable **augmente** la probabilité de réadmission.  
- **β < 0** → la variable **réduit** la probabilité de réadmission.  
- **exp(β)** = *odds ratio (OR)* : impact multiplicatif sur les chances de réadmission.

**Exemples d’interprétation :**

| Variable | β | OR | Interprétation |
|-----------|---|----|----------------|
| `time_in_hospital` | +0.42 | 1.52 | Les séjours plus longs augmentent le risque de réadmission. |
| `num_lab_procedures` | +0.27 | 1.31 | Un nombre élevé d’examens traduit une pathologie plus lourde. |
| `age_[0-30)` | −0.68 | 0.51 | Les patients jeunes présentent un risque plus faible de réadmission. |

---

### 3.6. Perspectives

- 🔹 Tester des modèles **ensemblistes** (*Random Forest*, *XGBoost*) et un **réseau de neurones (MLP)** pour mesurer le gain lié aux non-linéarités.  
- 🔹 Améliorer la **calibration** via *Platt Scaling* ou *Isotonic Regression*.  
- 🔹 Intégrer un **coût clinique différencié** pour ajuster le seuil selon le risque acceptable de faux positifs.  
- 🔹 Déployer un **score de risque interprétable** via un tableau de bord (*SHAP*, *Streamlit*, ou *Gradio*) permettant une visualisation claire des facteurs de risque individuels.



