import torch

from src.encoders.dinov2 import FEATURE_DIM, DINOv2Encoder


def test_parameters_are_frozen():
    encoder = DINOv2Encoder()
    assert all(not p.requires_grad for p in encoder.parameters())


def test_stays_in_eval_mode_even_after_train_call():
    encoder = DINOv2Encoder()
    encoder.train()
    assert not encoder.training


def test_output_shape_and_no_grad_tracking():
    encoder = DINOv2Encoder()
    dummy_images = torch.rand(2, 3, 224, 224)

    features = encoder(dummy_images)

    assert features.shape == (2, FEATURE_DIM)
    assert not features.requires_grad
