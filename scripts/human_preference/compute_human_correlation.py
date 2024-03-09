import pandas as pd
import scipy.stats as stats
import numpy as np

# read data
df = pd.read_excel('data/human_preference/sampled_faithdial_2024.xlsx', index_col=0)
X = df.iloc[:, 5:9].values
y = df.iloc[:, 9].values

X_train, y_train = [], []
for i in range(len(X)):
    if y[i] == 0:
        y_train.append(1)
        y_train.append(0)
    elif y[i] == 1:
        y_train.append(0)
        y_train.append(1)
    else:
        y_train.append(1)
        y_train.append(1)
    X_train.append(X[i][:2])
    X_train.append(X[i][2:])
X_train = np.array(X_train)
y_train = np.array(y_train)

max_corr = 0
max_beta = 0
for beta in np.arange(0, 1, step=0.01):
    x = []
    for i in range(len(X_train)):
        score = beta * X_train[i,0] + (1-beta) * X_train[i,1]
        x.append(score)

    # Calculate Pearson correlation coefficient
    pearson_corr, p_value = stats.pointbiserialr(y_train, x)
    if pearson_corr >= max_corr:
        max_corr = pearson_corr
        max_beta = beta
    if pearson_corr > 0:
        print(f"============== Beta: {beta} ==============")
        # Print the results
        print("Pearson correlation coefficient:", pearson_corr)
        print("p-value:", p_value)
print('beta:', max_beta)
print('pearson_corr:', max_corr)