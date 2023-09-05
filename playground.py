import pandas as pd

import torch
from torch import nn
from torch import Tensor
import math
import configparser
from torch.utils.data import Dataset, DataLoader

from torch import cuda
import series_transformer as st

dataset = st.CustomDataSet('Avg_data.csv',window_length=128,prediction_window=7)

params = st.ParameterProvider("series.config")
t1 = st.Transformer(params)
device_id = torch.cuda.current_device()
t1.cuda(device_id)
train_dataset, test_dataset = dataset.getSets()
st.train_cuda(t1,train_dataset,device_id,epochs=100,verbose_delay=-1)
for i in range(7):
    st.output_and_show(t1,train_dataset[0][0],train_dataset[0][2],device_id,i)