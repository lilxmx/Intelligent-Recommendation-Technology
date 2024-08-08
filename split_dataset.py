import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/ml-100k/u5.base",sep='\t',names=['uid','iid','rating','timestamp'],usecols=[0,1,2,3],header=None)
print(len(data))
explicit_data, implicit_data = train_test_split(data, test_size=0.5, random_state=2024)
print((explicit_data))
explicit_data.to_csv('datasets/ml-100k/u5.base.implicit.copy', sep='\t', index=False,header=False)
implicit_data.to_csv('datasets/ml-100k/u5.base.explicit.copy', sep='\t', index=False,header=False)
