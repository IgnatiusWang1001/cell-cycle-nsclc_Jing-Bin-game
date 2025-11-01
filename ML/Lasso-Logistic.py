import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sqlalchemy.orm.path_registry import path_is_property
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)

gse_tra_df_path = "data/GSE30129_transfer.xlsx"
gse_ins_df_path = "data/GSE30129_installment.xlsx"

tacg_tra_df_path = "data/tacg_transfer.xlsx"
tacg_ins_df_path = "data/tacg_installment.xlsx"

gse_ins_df = pd.read_excel(gse_ins_df_path)
tacg_ins_df = pd.read_excel(tacg_ins_df_path)

gse_tra_df = pd.read_excel(gse_tra_df_path)
tacg_tra_df = pd.read_excel(tacg_tra_df_path)

tra_df = pd.concat([gse_tra_df, tacg_tra_df], ignore_index=True)
# tra_df.to_excel("data/transfer.xlsx")
ins_df = pd.concat([gse_ins_df, tacg_ins_df], ignore_index=True)
# ins_df.to_excel("data/install.xlsx")


tra_map = {
    "M0": 0,
    "M1": 1,
}
ins_map = {
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3,
}

tra_df['Type'] = tra_df['Type'].map(tra_map)
ins_df['Type'] = ins_df['Type'].map(ins_map)

print(tra_df.columns)
print(ins_df.columns)

gen_col = ['ATM', 'ATR', 'CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'CDK1',
           'CDK2', 'CDK4', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'RB1', 'TP53']

print("分期的相关数据-----------------------------")
x = tra_df[gen_col]
y = tra_df['Type']

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        max_iter=10000,
        tol=1e-3,
        random_state=0
    ))
])

# ====== 超参搜索：只调 C（正则强度），以 AUC 为目标 ======
param_grid = {
    'logreg__C': [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=1,        # Windows 最稳（避免并行坑）；环境允许再改为 -1
    refit=True,
    verbose=0
)
grid.fit(x, y)

best_pipe = grid.best_estimator_
print("Best params:", grid.best_params_)
print("CV best AUC :", f"{grid.best_score_:.6f}")

# ====== CV 准确率（使用最佳模型的配置）======
cv_acc = cross_val_score(best_pipe, x, y, scoring='accuracy', cv=cv, n_jobs=1)
print(f"CV Accuracy (mean±std): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# ====== 全量数据评估（AUC & 准确率）======
proba = best_pipe.predict_proba(x)[:, 1]
auc_full = roc_auc_score(y, proba)
y_pred = (proba >= 0.5).astype(int)  # 如果需要，可改用 Youden J / 业务阈值
acc_full = accuracy_score(y, y_pred)

print("Full-data AUC      :", f"{auc_full:.6f}")
print("Full-data Accuracy :", f"{acc_full:.4f}")

# ====== 系数（标准化空间）======
scaler: StandardScaler = best_pipe.named_steps['scaler']
lr: LogisticRegression  = best_pipe.named_steps['logreg']

beta  = lr.coef_.ravel()     # 标准化后的 β_j
beta0 = lr.intercept_[0]     # 截距

print("\n[Standardized coefficients]  Score = beta0 + Σ beta_j * z(x_j)")
for name, b in zip(gen_col, beta):
    print(f"{name}: beta={b:.6f}")
print("Intercept (beta0):", float(beta0))


print("转移的相关数据-----------------------------")
X = ins_df[gen_col].values
Y = ins_df['Type'].astype(int)
# ---- 基础管道（scaler 可在网格里替换）----
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        max_iter=20000,    # 足够大，缓解不收敛
        tol=1e-1
    ))
])

# ===== 评分器：多分类 AUC(OVR, weighted) =====
auc_ovr_w = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')

# ===== 参数网格（关键：elasticnet 分支才给 l1_ratio）=====
param_grid = [
    # A) lbfgs + L2（稳健基线）
    {
        'scaler': [StandardScaler()],
        'logreg__solver': ['lbfgs'],
        'logreg__penalty': ['l2'],
        'logreg__C': [0.01, 0.1, 1.0, 3.0, 10.0],
    },
    # B) saga + L1/L2（无 l1_ratio）
    {
        'scaler': [StandardScaler()],
        'logreg__solver': ['saga'],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__C': [0.01, 0.1, 1.0, 3.0],
    },
    # C) saga + ElasticNet（这里只有 l1_ratio）
    {
        'scaler': [StandardScaler()],
        'logreg__solver': ['saga'],
        'logreg__penalty': ['elasticnet'],
        'logreg__C': [0.01, 0.1, 1.0, 3.0],
        'logreg__l1_ratio': [0.1, 0.5, 0.9],
    }
]

# ===== 交叉验证与搜索（Windows：n_jobs=1 最稳）=====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=auc_ovr_w,
    cv=cv,
    n_jobs=1,        # 避免 loky 并行问题；环境允许再改 -1
    refit=True,
    verbose=0
)
grid.fit(X, Y)

print("Best AUC(OVR, weighted):", f"{grid.best_score_:.6f}")
print("Best params:", grid.best_params_)

# ===== 最佳模型与评估 =====
best_pipe = grid.best_estimator_
P = best_pipe.predict_proba(X)
y_pred = best_pipe.predict(X)

classes_order = best_pipe.named_steps['logreg'].classes_
target_names = list(map(str, classes_order))  # 修复 classification_report 报错

auc_macro    = roc_auc_score(Y, P, multi_class='ovr', average='macro')
auc_weighted = roc_auc_score(Y, P, multi_class='ovr', average='weighted')
print(f"AUC (macro):    {auc_macro:.6f}")
print(f"AUC (weighted): {auc_weighted:.6f}")


# ===== 可部署系数（原始量纲）=====
scaler = best_pipe.named_steps['scaler']
lr     = best_pipe.named_steps['logreg']
B      = lr.coef_          # (K, p)，第 k 行 ↔ classes_order[k]
b0     = lr.intercept_     # (K,)

# 还原到原始特征量纲
B_orig  = B / scaler.scale_
b0_orig = b0 - (B * (scaler.mean_ / scaler.scale_)).sum(axis=1)

# 打印每类截距与前若干特征系数
p_preview = min(10, B.shape[1])
for k, ck in enumerate(classes_order):
    print(f"\n[Deploy Class={ck}] Intercept: {b0_orig[k]:.6f}")
    for j, g in enumerate(gen_col[:p_preview]):
        print(f"  {g}: {B_orig[k, j]:.6f}")