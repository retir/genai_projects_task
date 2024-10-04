import random
import torch


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)