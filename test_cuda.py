import torch

print(torch.cuda.is_available() )

x = torch.cuda.current_device() #returns you the ID of your current device
print(torch.cuda.get_device_name(x)) #returns you the name of the device

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

