import pandas as pd

gene_path = "data/3.基因列表.xlsx"

tacg_gene_path = "data/1.表达矩阵.xlsx"
tacg_installment_path = "data/2.特定列表分期.xlsx"
tacg_transfer_path = "data/2.特定列表转移.xlsx"


gene_df = pd.read_excel(gene_path)
tacg_gene_df = pd.read_excel(tacg_gene_path)
tacg_installment_df = pd.read_excel(tacg_installment_path)
tacg_transfer_df = pd.read_excel(tacg_transfer_path)


tacg_gene_New_df = (
    tacg_gene_df[tacg_gene_df['ID'].isin(gene_df['gene'])]
    .drop(columns='序号', errors='ignore')
    .set_index('ID')
    .T
)
tacg_gene_New_df = tacg_gene_New_df.reset_index().rename(columns={'index': 'user'})
tacg_gene_New_df.columns.name = None

tacg_installment_df_ = tacg_gene_New_df.merge(
    tacg_installment_df,
    left_on='user',
    right_on='ID',
    how='inner'
)

tacg_installment_df_.drop(columns='ID', inplace=True)
tacg_installment_df_.to_excel("data/tacg_installment.xlsx", index=False)
print(tacg_installment_df_)

tacg_transfer_df_ = tacg_gene_New_df.merge(
    tacg_transfer_df,
    left_on='user',
    right_on='ID',
    how='inner'
)

tacg_transfer_df_.drop(columns='ID', inplace=True)
tacg_transfer_df_.to_excel("data/tacg_transfer.xlsx", index=False)
print(tacg_transfer_df_)


