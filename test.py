import torch

def check_cuda_pytorch():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")

check_cuda_pytorch()
