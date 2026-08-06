import torch

print("=" * 40)
print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name        :", torch.cuda.get_device_name(0))
    print("CUDA Version    :", torch.version.cuda)
else:
    print("No GPU detected!")
print("=" * 40)