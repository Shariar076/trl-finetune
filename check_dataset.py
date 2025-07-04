import pandas as pd

ft_df = pd.read_csv("llm_consistent_conv_ft_data.csv")
print(ft_df.groupby('candidate').count())