import pandas as pd
import feature_eng as feat_eng

pd.set_option('display.max_columns', None)
data = pd.read_csv('datasets/sao-paulo-properties-april-2019.csv')

for col in data.columns:
    # print(col, '-', data[col].nunique())
    feat_eng.diagnostic_plots(data, col)
