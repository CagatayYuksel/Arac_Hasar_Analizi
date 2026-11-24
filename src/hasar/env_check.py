import ultralytics
import torch
import sys
import typing 

def check_environment() -> dict[str, str | bool]:
    """
    Grekli paketler için environment kontrolu ve CUDA bilgilerini kontrol eder"""
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    return info