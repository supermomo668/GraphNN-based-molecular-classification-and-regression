import pandas as pd, math
#
#from sklearn.model_selection import KFold

def make_splits(df_data, test_size=0.2, random_state:int = 101):
  """
  make as closely to equal amount of splits based on the hold-out size
  """
  # shuffle
  df_data = df_data.sample(frac=1, random_state=101)
  # split data into n
  diff = int(len(df_data)*test_size)
  k = len(df_data)//diff
  df_sp = [df_data.iloc[i: i+diff] for i in range(k-1)]
  return df_sp + [df_data.iloc[(k-1)*diff: len(df_data)]]


