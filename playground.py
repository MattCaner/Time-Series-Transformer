import pandas as pd

import torch
from torch import nn
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader

from torch import cuda
import series_transformer as st

file = open('Avg_data.csv','r')
df = pd.read_csv(file)

df[["day", "month", "year"]] = df["Date"].str.split("/", expand = True)
df['day'] = df['day'].astype(float)
df['month'] = df['month'].astype(float)
df['year'] = df['year'].astype(float)

df = df.drop(['Date'],axis=1)

data = torch.tensor(df.values)

dataset = st.CustomDataSet('Avg_data.csv')


params = st.ParameterProvider("params_series.config")
t1 = st.Transformer(params)

print(dataset[0][0][-1])
print(dataset[0][1][0])
print('---')
print(dataset[0][1][1])
print(dataset[0][2][0])

torch.unsqueeze(dataset[0][0][-1],0)

print(torch.cuda.is_available())
device_id = torch.cuda.current_device()

t1.cuda(device_id)

output = t1(torch.unsqueeze(dataset[0][0],0).to(torch.float).cuda(device_id), torch.unsqueeze(dataset[0][1],0).to(torch.float).cuda(device_id))