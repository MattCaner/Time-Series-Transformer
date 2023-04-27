import transformer as t
from torch import cuda

general_params = t.ParameterProvider("params_local.config")

v_in = t.VocabProvider(general_params,general_params.provide("language_in_file"))
v_out = t.VocabProvider(general_params,general_params.provide("language_out_file"))
cd = t.CustomDataSet(general_params.provide("language_in_file"), general_params.provide("language_out_file"),v_in,v_out)
train_dataset, test_dataset = cd.getSets()
test_mode = False
if test_mode:
    test_dataset = train_dataset

tt = t.Transformer(general_params,v_in,v_out,True)

print("starting training")

t.train_cuda(tt,cd,epochs=5,device=cuda.current_device())
