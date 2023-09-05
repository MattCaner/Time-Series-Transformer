import series_transformer as st

dataset = st.CustomDataSet('data_agg.csv',window_length=128,prediction_window=7,require_date_split = False, drop_idx_column = True)

params = st.ParameterProvider("series_senti.config")
t1 = st.Transformer(params)

device_id = torch.cuda.current_device()
t1.cuda(device_id)