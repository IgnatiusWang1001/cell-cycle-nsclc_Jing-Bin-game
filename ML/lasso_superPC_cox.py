# ===== 生成虚拟数据，用于验证 lasso_superpc_cox =====
import numpy as np
import pandas as pd
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import get_status
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter


tcga_time_df_path = "data/TCGA生存时间与随访时间.xlsx"
tcga_time_df = pd.read_excel(tcga_time_df_path)
tcga_time_df['event'] = tcga_time_df['demographic.vital_status'].replace({"Alive": 0, "Dead": 1})
tcga_time_df['time'] = pd.to_numeric(tcga_time_df['follow_ups.days_to_follow_up'], errors='coerce')
tcga_time_df = tcga_time_df.dropna(subset=['time']).copy()
tcga_time_df = tcga_time_df.drop(columns=['demographic.vital_status', 'demographic.days_to_death',
                                          'days_to_last_followup', 'follow_ups.days_to_follow_up'])

tcga_gene_path = "data/tacg_transfer.xlsx"
tcga_gene_df = pd.read_excel(tcga_gene_path)
tcga_gene_df = tcga_gene_df.drop(columns=['Type'])
tcga_gene_df['user'] = tcga_gene_df['user'].astype(str).str[:12]

tcga_df = tcga_gene_df.merge(
    tcga_time_df,
    left_on="user",
    right_on="ID",
    how="inner",
)
tcga_df = tcga_df.drop(columns=['ID', 'user'])
tcga_gene_columns = ['ATM', 'ATR', 'CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'CDK1',
                     'CDK2', 'CDK4', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'RB1', 'TP53']


gse_time_path = "data/GSE30219生存随访.xlsx"
gse_time_df = pd.read_excel(gse_time_path)
gse_time_df['event'] = gse_time_df['event'].replace({" ALIVE": 0, " DEAD": 1})
gse_time_df = gse_time_df[gse_time_df['event'] != ' NTL']
gse_time_df['time'] = pd.to_numeric(gse_time_df['!Sample_characteristics_ch1'], errors='coerce')
gse_time_df = gse_time_df.dropna(subset=['time']).copy()
gse_time_df = gse_time_df.drop(columns=['!Sample_characteristics_ch1'])

gse_gene_path = "data/GSE30129_transfer.xlsx"
gse_gene_df = pd.read_excel(gse_gene_path)
# gse_gene_df['user'] = gse_gene_df['user'].astype(str).str[:12]
gse_gene_df = gse_gene_df.drop(columns=['Type'])
gse_df = gse_gene_df.merge(
    gse_time_df,
    left_on="user",
    right_on="ID",
    how="inner",
)
gse_df = gse_df.drop(columns=['ID', 'user'])
gse_gene_columns = ['ATM', 'ATR', 'CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'CDK1',
                    'CDK2', 'CDK4', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'RB1', 'TP53'
                    ]


def lasso_superpc_cox(df, time_col, event_col, gene_cols, alpha=None, n_pc=1):
    """
    df: 包含生存数据和基因表达的 DataFrame
    time_col: 生存时间列名
    event_col: 事件列名
    gene_cols: 基因列名列表
    alpha: Lasso 正则化强度 (None 用 CV 自动选择)
    n_pc: 取的主成分数
    """
    X = df[gene_cols].values
    y_time = df[time_col].values
    y_event = df[event_col].values

    # 标准化
    X_scaled = StandardScaler().fit_transform(X)

    # Lasso 特征选择（用生存时间做回归选择，仅为示范；实战可改为与风险/打分相关的监督信号）
    if alpha is None:
        lasso = LassoCV(cv=5, max_iter=5000).fit(X_scaled, y_time)
    else:
        lasso = Lasso(alpha=alpha, max_iter=5000).fit(X_scaled, y_time)

    selected_idx = np.where(lasso.coef_ != 0)[0]
    if len(selected_idx) == 0:
        raise ValueError("Lasso 没有选择到任何基因，请调低 alpha 或放宽选择策略。")

    X_selected = X_scaled[:, selected_idx]

    # SuperPC（PCA提取主成分）
    pca = PCA(n_components=n_pc)
    pc_scores = pca.fit_transform(X_selected)
    df_pc = pd.DataFrame(pc_scores, columns=[f"PC{i + 1}" for i in range(n_pc)], index=df.index)

    # 单变量 Cox 回归（用 PC1）
    df_cox = pd.concat([df[[time_col, event_col]], df_pc.iloc[:, [0]]], axis=1)
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)

    hr = np.exp(cph.params_[0])
    ci = cph.confidence_intervals_.iloc[0].apply(np.exp)
    pval = cph.summary.loc[df_pc.columns[0], "p"]

    return {
        "HR": hr,
        "lower_CI": ci[0],
        "upper_CI": ci[1],
        "pval": pval,
        "selected_genes": [gene_cols[i] for i in selected_idx]
    }



tcga_result = lasso_superpc_cox(tcga_df, 'time', 'event', tcga_gene_columns, alpha=0.001, n_pc=1)
print(tcga_result)

gse_result = lasso_superpc_cox(gse_df, 'time', 'event', gse_gene_columns, alpha=0.001, n_pc=1)
print(gse_result)

tcga_result.pop('selected_genes', None)
tcga_result['Study'] = 'TCGA'
gse_result.pop('selected_genes', None)
gse_result['Study'] = 'GSE'


meta_df = pd.DataFrame([tcga_result, gse_result])

def _infer_logHR_SE(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一把每行的效果量转成 logHR 和 SE。
    优先使用: HR + 95%CI -> logHR, SE
    备选: 已给出 logHR + SE / var 列时直接使用
    """
    out = df.copy()
    if {"HR", "lower_CI", "upper_CI"}.issubset(out.columns):
        out["logHR"] = np.log(out["HR"])
        out["SE"] = (np.log(out["upper_CI"]) - np.log(out["lower_CI"])) / (2 * 1.96)
    elif {"logHR", "SE"}.issubset(out.columns):
        pass
    elif {"logHR", "var"}.issubset(out.columns):
        out["SE"] = np.sqrt(out["var"])
    else:
        raise ValueError("需要 (HR, lower_CI, upper_CI) 或 (logHR, SE/var) 列")
    return out


def meta_fixed_random(df: pd.DataFrame):
    """
    返回固定效应与随机效应（DerSimonian-Laird）的合并结果与异质性统计。
    """
    d = _infer_logHR_SE(df)
    w_fixed = 1.0 / (d["SE"] ** 2)
    mu_fixed = np.sum(w_fixed * d["logHR"]) / np.sum(w_fixed)
    se_fixed = np.sqrt(1.0 / np.sum(w_fixed))
    res_fixed = {
        "model": "fixed",
        "pooled_logHR": mu_fixed,
        "pooled_HR": float(np.exp(mu_fixed)),
        "lower_CI": float(np.exp(mu_fixed - 1.96 * se_fixed)),
        "upper_CI": float(np.exp(mu_fixed + 1.96 * se_fixed)),
        "SE": float(se_fixed),
    }

    # 异质性
    Q = float(np.sum(w_fixed * (d["logHR"] - mu_fixed) ** 2))
    df_q = len(d) - 1
    C = np.sum(w_fixed) - (np.sum(w_fixed ** 2) / np.sum(w_fixed))
    tau2 = max(0.0, (Q - df_q) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - df_q) / Q) * 100 if Q > 0 else 0.0

    # 随机效应（DL）
    w_rand = 1.0 / (d["SE"] ** 2 + tau2)
    mu_rand = np.sum(w_rand * d["logHR"]) / np.sum(w_rand)
    se_rand = np.sqrt(1.0 / np.sum(w_rand))
    res_random = {
        "model": "random(DL)",
        "pooled_logHR": mu_rand,
        "pooled_HR": float(np.exp(mu_rand)),
        "lower_CI": float(np.exp(mu_rand - 1.96 * se_rand)),
        "upper_CI": float(np.exp(mu_rand + 1.96 * se_rand)),
        "SE": float(se_rand),
    }

    hetero = {"Q": Q, "df": df_q, "I2_percent": I2, "tau2": tau2}
    # 同时返回各研究权重（便于森林图标注）
    weights = pd.DataFrame({
        "Study": df["Study"].values if "Study" in df.columns else np.arange(len(df)),
        "weight_fixed": (w_fixed / np.sum(w_fixed)).values,
        "weight_random": (w_rand / np.sum(w_rand)).values
    })
    return res_fixed, res_random, hetero, d, weights


fixed_res, random_res, hetero, d_used, weights = meta_fixed_random(meta_df)

print("固定效应合并结果:", fixed_res)
print("随机效应合并结果:", random_res)
print("异质性指标:", hetero)


# 画森林图
def forest_plot(df: pd.DataFrame, pooled: dict, title="Meta-analysis (HR)"):
    """
    df: 原始 df，至少包含 Study, HR, lower_CI, upper_CI
    pooled: 合并结果字典（pooled_HR, lower_CI, upper_CI）
    """
    df = df.copy()
    y = list(range(len(df), 0, -1))
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(df) + 2))

    # 各研究点估计与区间
    ax.errorbar(df["HR"], y,
                xerr=[df["HR"] - df["lower_CI"], df["upper_CI"] - df["HR"]],
                fmt='o', capsize=3)
    # 合并
    ax.errorbar([pooled["pooled_HR"]], [0],
                xerr=[[pooled["pooled_HR"] - pooled["lower_CI"]],
                      [pooled["upper_CI"] - pooled["pooled_HR"]]],
                fmt='s', capsize=5, label='Pooled')

    # 视觉设置
    ax.axvline(1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y + [0])
    names = df["Study"].tolist() if "Study" in df.columns else [f"Study_{i + 1}" for i in range(len(df))]
    ax.set_yticklabels(names + ["Pooled"])
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()


forest_plot(meta_df, random_res, title="Random-effects Meta-analysis (2 studies)")
