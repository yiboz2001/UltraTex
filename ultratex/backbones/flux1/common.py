import torch
from contextlib import contextmanager

@contextmanager
def nvtx(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

def nsys_start():
    if torch.cuda.is_available():
        # drain queued kernels first so the start marker is accurate
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

def nsys_stop():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
