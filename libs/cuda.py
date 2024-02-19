import torch


def is_cuda_available():
    return torch.cuda.is_available()


def current_device():
    if is_cuda_available():
        return torch.cuda.get_device_name(0)
    else:
        return "cpu"


def current_cuda_version():
    return torch.version.cuda
