import pytest
import torch
from model.mnist_net import MNISTNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds the limit of 20000"

def test_batch_norm_usage():
    model = MNISTNet()
    has_batch_norm = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"

def test_dropout_usage():
    model = MNISTNet()
    has_dropout = any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_fully_connected_layer():
    model = MNISTNet()
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert has_fc, "Model should use fully connected layers" 