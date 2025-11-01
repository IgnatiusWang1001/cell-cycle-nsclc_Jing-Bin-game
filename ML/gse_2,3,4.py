import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df_path = "data/GSE30129_transfer.xlsx"
df = pd.read_excel(df_path)
df['Type'] = df['Type'].replace({'M0': 0, "M1": 1})

scaler = MinMaxScaler()
x = df.drop(columns=['Type', 'user'])
X = scaler.fit_transform(x)
y = df['Type'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("准确率:", acc)

risk_scores = model.predict_proba(X)[:, 1]
print(risk_scores)
risk_scores_df = pd.DataFrame({
    "risk_scores": risk_scores
})
risk_scores_df = pd.concat([risk_scores_df, x], axis=1)
risk_scores_df.to_excel("out_data/GSE30129_risk_scores.xlsx")
# 计算的风险评分与不同队列患者的关系
risk_scores = model.predict_proba(x_test)[:, 1]
train_risk_scores = model.predict_proba(x_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, 1 - risk_scores)
auc = roc_auc_score(y_test, 1 - risk_scores)
train_fpr, train_tpr, _ = roc_curve(y_train, train_risk_scores)
train_auc = roc_auc_score(y_train, train_risk_scores)

plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, color='y', label='Train ROC curve (AUC = %0.2f)' % train_auc)
plt.plot(fpr, tpr, color='blue', label='Test ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

gse_train_roc_df = pd.DataFrame({
    "fpr": train_fpr,
    "tpr": train_tpr,
})
gse_train_roc_df.to_excel(f"out_data/gse_train_roc_auc({train_auc}).xlsx")
gse_test_roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
})
gse_test_roc_df.to_excel(f"out_data/gse_test_roc_auc({auc}).xlsx")
