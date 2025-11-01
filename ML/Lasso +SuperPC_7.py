import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df_path = "data/tacg_transfer.xlsx"
df = pd.read_excel(df_path)

scaler = MinMaxScaler()
df['Type'] = df['Type'].replace({"M0": 0, "M1": 1})
x = df.drop(columns=['user', 'Type'])
X = scaler.fit_transform(x)
y = df['Type'].values

lasso = Lasso(alpha=0.001)
lasso.fit(X, y)

select_features = np.where(lasso.coef_ != 0)[0]
x_select = X[:, select_features]

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_select)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=12)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_risk = model.predict_proba(x_test)[:, 1]

c_index = concordance_index(y_test, y_pred_risk)
print("tcag转移的C指数", c_index)

# ---------------------------------转移——————————————————————————————
df_path = "data/tacg_installment.xlsx"
df = pd.read_excel(df_path)

scaler = MinMaxScaler()
df['Type'] = df['Type'].replace({"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3})
x = df.drop(columns=['user', 'Type'])
X = scaler.fit_transform(x)
y = df['Type'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='liblinear')
model.fit(x_train, y_train)
y_pred_prob = model.predict_proba(x_test)


def _cindex_from_scores(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    concordant = 0
    ties = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            si, sj = scores[i], scores[j]
            if (si > sj and y_true[i] > y_true[j]) or (sj > si and y_true[j] > y_true[i]):
                concordant += 1
            elif si == sj:
                ties += 1
    return (concordant + 0.5 * ties) / total if total > 0 else np.nan

def multiclass_cindex(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        class_vals = np.arange(y_pred.shape[1], dtype=float)  # 0..K-1
        scores = (y_pred * class_vals).sum(axis=1)
    else:
        scores = y_pred

    return _cindex_from_scores(y_true, scores)

c_index = multiclass_cindex(y_test, y_pred_prob)
print("tcag分期的C指数:", c_index)

#-----------------------------------------------------------
#-------------------------------------------------------------

df_path = "data/GSE30129_transfer.xlsx"
df = pd.read_excel(df_path)

scaler = MinMaxScaler()
df['Type'] = df['Type'].replace({"M0": 0, "M1": 1})
x = df.drop(columns=['user', 'Type'])
X = scaler.fit_transform(x)
y = df['Type'].values

lasso = Lasso(alpha=0.001)
lasso.fit(X, y)

select_features = np.where(lasso.coef_ != 0)[0]
x_select = X[:, select_features]

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_select)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=12)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_risk = model.predict_proba(x_test)[:, 1]

c_index = concordance_index(y_test, y_pred_risk)
print("gse转移的C指数", c_index)

# ---------------------------------转移——————————————————————————————
df_path = "data/GSE30129_installment.xlsx"
df = pd.read_excel(df_path)

scaler = MinMaxScaler()
df['Type'] = df['Type'].replace({"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3})
x = df.drop(columns=['user', 'Type'])
X = scaler.fit_transform(x)
y = df['Type'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='liblinear')
model.fit(x_train, y_train)
y_pred_prob = model.predict_proba(x_test)


def _cindex_from_scores(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    concordant = 0
    ties = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            si, sj = scores[i], scores[j]
            if (si > sj and y_true[i] > y_true[j]) or (sj > si and y_true[j] > y_true[i]):
                concordant += 1
            elif si == sj:
                ties += 1
    return (concordant + 0.5 * ties) / total if total > 0 else np.nan

def multiclass_cindex(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        class_vals = np.arange(y_pred.shape[1], dtype=float)  # 0..K-1
        scores = (y_pred * class_vals).sum(axis=1)
    else:
        scores = y_pred

    return _cindex_from_scores(y_true, scores)

c_index = multiclass_cindex(y_test, y_pred_prob)
print("gse分期的C指数:", c_index)