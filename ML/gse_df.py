import pandas as pd

gene_path = "data/3.基因列表.xlsx"

gse30219_gene_path = "data/GSE30219-表达矩阵.csv"
gse30219_installment_path = "data/GSE30219-特定列表 - 分期.xlsx"
gse30129_transfer_path = "data/GSE30219-特定列表 - 转移.xlsx"


gene_df = pd.read_excel(gene_path)
gse30219_gene_df = pd.read_csv(gse30219_gene_path)
gse30219_installment_df = pd.read_excel(gse30219_installment_path)
gse30129_transfer_df = pd.read_excel(gse30129_transfer_path)


gse30219_gene_New_df = (
    gse30219_gene_df[gse30219_gene_df['ID'].isin(gene_df['gene'])]
    .drop(columns='序号', errors='ignore')
    .set_index('ID')
    .T
)
gse30219_gene_New_df = gse30219_gene_New_df.reset_index().rename(columns={'index': 'user'})
gse30219_gene_New_df.columns.name = None

gse30219_installment_df_ = gse30219_gene_New_df.merge(
    gse30219_installment_df,
    left_on='user',
    right_on='ID',
    how='inner'
)

gse30219_installment_df_.drop(columns='ID', inplace=True)
gse30219_installment_df_.to_excel("data/GSE30129_installment.xlsx", index=False)
print(gse30219_installment_df_)

gse30219_transfer_df_ = gse30219_gene_New_df.merge(
    gse30129_transfer_df,
    left_on='user',
    right_on='ID',
    how='inner'
)

gse30219_transfer_df_.drop(columns='ID', inplace=True)
gse30219_transfer_df_.to_excel("data/GSE30129_transfer.xlsx", index=False)
print(gse30219_transfer_df_)


