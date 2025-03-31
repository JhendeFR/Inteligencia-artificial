import torch
print("¿CUDA está disponible?:", torch.cuda.is_available())
print("¿GPU actual?:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No disponible")