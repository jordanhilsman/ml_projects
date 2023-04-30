import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
np.random.seed(42)
df = pd.read_csv(r"C:\Users\Jordan\Downloads\fetal_health.csv")

miss_values = df.columns[df.isnull().any()]
print(f"Missing values:\n{df[miss_values].isnull().sum()}")

null_values = df.columns[df.isna().any()]
print(f"Null values:\n{df[null_values].isna().sum()}")

print(df['fetal_health'].value_counts())




# 1 (normal): 1655, 2 (suspect): 295, 3 (pathological): 176
normal = 1655
suspect = 295
pathological = 176
# imbalanced dataset, so classification accuracy is a misguiding metric

correlation = df.corr()
plt.figure(figsize=
           (25,25))
sns.heatmap(correlation,annot=True)
plt.show()

# look at box plots for the most strong correlations (acceleration, prolonged decelerations,
# abnormal st variablity, % with abnorm, mean value of lt var
interesting = ['accelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability']

for item in interesting:
    sns.boxplot(x=df["fetal_health"], y=df[item])
    plt.tight_layout()
    plt.show()



#from this, we see we should drop histogram mean, mode, median, as they are very collinear

columns = df.columns.tolist()[:-1]
scale_X = StandardScaler()
#data must be standardized due to difference in magnitude across features
X = pd.DataFrame(scale_X.fit_transform(df.drop(['fetal_health'], axis=1),), columns=columns)
y = df['fetal_health']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42, stratify=y)

#Models
logistic_regression = LogisticRegression(random_state=42, max_iter=200)
LR_fitted = logistic_regression.fit(X_train, y_train)

KNeighbors = KNeighborsClassifier()
KN_fitted = KNeighbors.fit(X_train, y_train)

RandForest = RandomForestClassifier()
RF_fitted = RandForest.fit(X_train, y_train)

NaiveBayes = GaussianNB()
NB_fitted = NaiveBayes.fit(X_train, y_train)

# scoring dictionary for cross_val_score
scores ={
    'accuracy' : make_scorer(accuracy_score),
    'precision' : make_scorer(precision_score,average='weighted'),
    'recall' : make_scorer(recall_score,average='weighted'),
    'f1_score' : make_scorer(f1_score,average='weighted')
}


kfold = KFold(n_splits=10, random_state=42, shuffle=True)
models = [LR_fitted, KN_fitted, RF_fitted, NB_fitted]
for model in models:
    i = 0
    for score in scores.values():
        train_res = cross_val_score(estimator=model,
                                          X=X_train,
                                          y=y_train,
                                          cv=kfold,
                                          scoring=score)

        test_res = cross_val_score(estimator=model,
                                          X=X_test,
                                          y=y_test,
                                          cv=kfold,
                                          scoring=score)
        print(f"The {list(scores.keys())[i]} for the model {model} on the training set is {np.mean(train_res):.4f}")
        print(f"The {list(scores.keys())[i]} for the model {model} on the testing set is {np.mean(test_res):.4f}")
        i += 1


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for model, ax in zip(models, axes.flatten()):
    ConfusionMatrixDisplay.from_estimator(model,
                          X_test,
                          y_test,
                          ax=ax,
                          cmap='Blues',
                         display_labels=['normal', 'suspect', 'pathological'])
    ax.title.set_text(type(model).__name__)
plt.tight_layout()
plt.show()


