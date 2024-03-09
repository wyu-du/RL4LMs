import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV


# read data
df = pd.read_excel('data/human_preference/sampled_multidoc2dial_2024.xlsx', index_col=0)
X = np.array(df.iloc[:, 5:9].values, dtype=float)
y = np.array(df.iloc[:, 9].values, dtype=np.compat.long)

X_train, y_train = [], []
for i in range(len(X)):
    X_train.append(X[i][:2])
    X_train.append(X[i][2:])
    if y[i] == 0:
        y_train.append(1)
        y_train.append(0)
    elif y[i] == 1:
        y_train.append(0)
        y_train.append(1)
    else:
        y_train.append(0)
        y_train.append(0)
X_train = np.array(X_train)
y_train = np.array(y_train)

# param_grid = {
#     'solver': ['liblinear', 'lbfgs'],
#     'penalty': ['l1', 'l2'],
#     'C': [0.01, 0.1, 1, 10, 100],
#     'class_weight': [None, 'balanced']
# }

# grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)

clf = LogisticRegression(C=1, random_state=42, penalty='l1', solver='liblinear', max_iter=1000).fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.get_params())
# print(clf.coef_)
# print(clf.intercept_)
# print('input:', X_train[0])
print('prob:', clf.predict_proba(X_train)[0])
# print('label:', y_train[0])
# print('input:', X_train[1])
# print('prob:', clf.predict_proba(X_train)[1])
# print('label:', y_train[1])
# print(clf.predict_proba(X_train))

# save model
with open('./ckpts/mdd_human_comparison.sav', 'wb') as f:
    pickle.dump(clf, f)
# load model
with open('./ckpts/mdd_human_comparison.sav', 'rb') as f:
    model = pickle.load(f)
print(model.predict_proba(X_train[0].reshape(1, -1)))