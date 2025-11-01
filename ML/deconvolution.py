"""
TCGA队列中计算的风险评分与不同方法反卷积的关系
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.decomposition import NMF, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.optimize import nnls


def load_bulk_expression(
        excel_path: str,
        drop_first_col: bool = True,
        drop_last_col_as_label: bool = True,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    if drop_first_col and df.shape[1] >= 2:
        df = df.iloc[:, 1:]
    if drop_last_col_as_label and df.shape[1] >= 2:
        df = df.iloc[:, :-1]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[df < 0] = 0.0
    if df.index.name is None:
        df.index.name = "Sample"
    return df


def row_normalize_to_one(W: np.ndarray) -> np.ndarray:
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def nmf_reconstruction_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.linalg.norm(X - W @ H, ord='fro'))


def nmf_consensus_stability(W_list: List[np.ndarray]) -> float:
    if len(W_list) == 0:
        return 0.0
    n_samples = W_list[0].shape[0]
    C = np.zeros((n_samples, n_samples), dtype=float)
    for W in W_list:
        labels = np.argmax(W, axis=1)
        for i in range(n_samples):
            C[i, labels == labels[i]] += 1.0
    C /= len(W_list)

    mask = ~np.eye(n_samples, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float(C[mask].mean())


def nmf_model_selection(
        X: np.ndarray,
        K_range: range = range(2, 11),
        n_repeats: int = 10,
        max_iter: int = 1000,
        random_state_base: int = 123,
) -> pd.DataFrame:
    rows = []
    for K in K_range:
        W_list, H_list, errs = [], [], []
        for r in range(n_repeats):
            nmf = NMF(n_components=K, init='nndsvda', max_iter=max_iter, random_state=random_state_base + r)
            W = nmf.fit_transform(X)
            H = nmf.components_
            Wn = row_normalize_to_one(W)
            err = nmf_reconstruction_error(X, W, H)
            W_list.append(Wn)
            H_list.append(H)
            errs.append(err)
        stability = nmf_consensus_stability(W_list)
        rows.append({
            'K': K,
            'recon_error_mean': float(np.mean(errs)),
            'recon_error_std': float(np.std(errs)),
            'stability': stability,
        })
    df = pd.DataFrame(rows)

    k_max_stab = df.loc[df['stability'].idxmax(), 'K']

    errs = df.set_index('K')['recon_error_mean']
    rel_drops = errs.shift(1) - errs
    rel_drops = rel_drops.fillna(0)

    initial_drop = (errs.iloc[0] - errs.min()) if len(errs) > 1 else 0
    elbow_K = None
    if initial_drop > 0:
        for K in df['K']:
            if rel_drops.get(K, 0) / initial_drop < 0.05 and K >= 3:
                elbow_K = K
                break
    chosen_K = int(k_max_stab if elbow_K is None else min(k_max_stab, elbow_K))
    df['chosen'] = df['K'] == chosen_K
    return df


def nmf_fit(X: np.ndarray, K: int, random_state: int = 42, max_iter: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    nmf = NMF(n_components=K, init='nndsvda', max_iter=max_iter, random_state=random_state)
    W = nmf.fit_transform(X)
    H = nmf.components_
    Wn = row_normalize_to_one(W)
    return Wn, H


def ica_deconvolution(X: np.ndarray, K: int, random_state: int = 42, max_iter: int = 1000) -> Tuple[
    np.ndarray, np.ndarray]:
    ica = FastICA(n_components=K, random_state=random_state, max_iter=max_iter)
    X_centered = X - X.mean(axis=0, keepdims=True)
    S_sources = ica.fit_transform(X_centered)
    A_mixing = ica.mixing_

    S = np.abs(A_mixing.T)

    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)

    n_samples = X.shape[0]
    W = np.zeros((n_samples, K))
    for i in range(n_samples):
        b = X[i, :]
        w, _ = nnls(S.T, b)  # S.T: genes x K
        W[i, :] = w
    W = row_normalize_to_one(W)
    return W, S


# ---------------------------- PCA+KMeans Baseline ----------------------------

def pcaproject_deconvolution(X: np.ndarray, K: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, G = X.shape

    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    km.fit(X.T)
    labels = km.labels_

    S = np.zeros((K, G), dtype=float)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue

        S[k, idx] = 1.0 / len(idx)

    row_norm = np.linalg.norm(S, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    S = S / row_norm

    W = np.zeros((n_samples, K), dtype=float)
    ST = S.T  # genes x K
    for i in range(n_samples):
        b = X[i, :]
        w, _ = nnls(ST, b)
        W[i, :] = w

    W_sum = W.sum(axis=1, keepdims=True)
    W_sum[W_sum == 0] = 1.0
    W = W / W_sum

    return W, S


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_matrix_excel(df: pd.DataFrame, path: str):
    df.to_excel(path)


def run_pipeline(
        bulk_path: str,
        out_dir: str,
        drop_first_col: bool = True,
        drop_last_col_as_label: bool = True,
        k_min: int = 2,
        k_max: int = 10,
        nmf_repeats: int = 10,
):
    ensure_outdir(out_dir)
    bulk_df = load_bulk_expression(bulk_path, drop_first_col, drop_last_col_as_label)

    X = bulk_df.values.astype(float)

    sel_df = nmf_model_selection(X, K_range=range(k_min, k_max + 1), n_repeats=nmf_repeats)
    sel_df.to_csv(os.path.join(out_dir, 'nmf_model_selection.csv'), index=False)
    chosen_K = int(sel_df.loc[sel_df['chosen'], 'K'].iloc[0])

    W_nmf, H_nmf = nmf_fit(X, K=chosen_K)
    nmf_W_df = pd.DataFrame(W_nmf, index=bulk_df.index, columns=[f"comp_{i + 1}" for i in range(chosen_K)])
    nmf_H_df = pd.DataFrame(H_nmf, index=[f"comp_{i + 1}" for i in range(chosen_K)], columns=bulk_df.columns)
    nmf_W_df.to_excel(os.path.join(out_dir, 'nmf_proportions.xlsx'))
    nmf_H_df.to_excel(os.path.join(out_dir, 'nmf_signatures.xlsx'))

    W_ica, S_ica = ica_deconvolution(X, K=chosen_K)
    ica_W_df = pd.DataFrame(W_ica, index=bulk_df.index, columns=[f"comp_{i + 1}" for i in range(chosen_K)])
    ica_S_df = pd.DataFrame(S_ica, index=[f"comp_{i + 1}" for i in range(chosen_K)], columns=bulk_df.columns)
    ica_W_df.to_excel(os.path.join(out_dir, 'ica_proportions.xlsx'))
    ica_S_df.to_excel(os.path.join(out_dir, 'ica_signatures.xlsx'))

    W_pca, S_pca = pcaproject_deconvolution(X, K=chosen_K)
    pca_W_df = pd.DataFrame(W_pca, index=bulk_df.index, columns=[f"comp_{i + 1}" for i in range(chosen_K)])
    pca_S_df = pd.DataFrame(S_pca, index=[f"comp_{i + 1}" for i in range(chosen_K)], columns=bulk_df.columns)
    pca_W_df.to_excel(os.path.join(out_dir, 'pcaproject_proportions.xlsx'))
    pca_S_df.to_excel(os.path.join(out_dir, 'pcaproject_signatures.xlsx'))

    for name, df in {
        'nmf': nmf_W_df,
        'ica': ica_W_df,
        'pcaproject': pca_W_df,
    }.items():
        long_df = df.copy()
        long_df['Sample'] = long_df.index
        long_df = long_df.melt(id_vars='Sample', var_name='Component', value_name='Proportion')
        long_df.to_csv(os.path.join(out_dir, f'{name}_proportions_long.csv'), index=False)

    print(
        f"Done. Chosen K={chosen_K}. Files written to: \n"
        f" - {os.path.join(out_dir, 'nmf_model_selection.csv')}\n"
        f" - {os.path.join(out_dir, 'nmf_proportions.xlsx')}\n"
        f" - {os.path.join(out_dir, 'nmf_signatures.xlsx')}\n"
        f" - {os.path.join(out_dir, 'ica_proportions.xlsx')}\n"
        f" - {os.path.join(out_dir, 'ica_signatures.xlsx')}\n"
        f" - {os.path.join(out_dir, 'pcaproject_proportions.xlsx')}\n"
        f" - {os.path.join(out_dir, 'pcaproject_signatures.xlsx')}\n"
    )


if __name__ == "__main__":
    BULK_PATH = r"data\\tacg_transfer.xlsx"
    OUT_DIR = r"out_data"

    run_pipeline(
        bulk_path=BULK_PATH,
        out_dir=OUT_DIR,
        drop_first_col=True,
        drop_last_col_as_label=True,
        k_min=2,
        k_max=10,
        nmf_repeats=10,
    )
