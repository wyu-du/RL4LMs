import pandas as pd
import scipy.stats as stats
import numpy as np

# read data
df = pd.read_excel('data/human_preference/sampled_faithdial.xlsx', index_col=0)
X = df.iloc[:, 8:12].values
y = df.iloc[:, 12].values

# for beta in np.arange(0, 1, step=0.01):
for beta in [0.85]:
    x = []
    for i in range(len(X)):
        score_1 = beta * X[i,0] + (1-beta) * X[i,1]
        score_2 = beta * X[i,2] + (1-beta) * X[i,3]
        if score_1 > score_2:
            x.append(0)
        elif score_1 < score_2:
            x.append(1)
        else:
            x.append(2)

    # Calculate Pearson correlation coefficient
    pearson_corr, p_value = stats.pearsonr(x, y)
    # if pearson_corr > 0:
    print(f"============== Beta: {beta} ==============")
    print("Acc:", (x == y).mean())
    # Print the results
    print("Pearson correlation coefficient:", pearson_corr)
    print("p-value:", p_value)