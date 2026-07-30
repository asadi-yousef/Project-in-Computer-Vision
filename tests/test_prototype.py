import pytest
import torch

from src.classifiers.prototype import compute_class_prototypes, predict_by_cosine_similarity


def test_prototypes_are_l2_normalized():
    features = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]])
    labels = torch.tensor([0, 0, 1])

    prototypes = compute_class_prototypes(features, labels, num_classes=2)

    assert torch.allclose(prototypes.norm(dim=1), torch.ones(2), atol=1e-6)


def test_prototype_matches_hand_computed_value():
    # (3,4) and (6,8) both normalize to (0.6,0.8); their mean is already
    # unit-norm, so class 0's prototype should be exactly (0.6, 0.8).
    # (0,5) normalizes to (0,1) -> class 1's prototype is (0,1).
    features = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 5.0]])
    labels = torch.tensor([0, 0, 1])

    prototypes = compute_class_prototypes(features, labels, num_classes=2)

    assert torch.allclose(prototypes[0], torch.tensor([0.6, 0.8]), atol=1e-5)
    assert torch.allclose(prototypes[1], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_raises_for_class_with_no_features():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0])  # class 1 never appears

    with pytest.raises(ValueError, match="Class 1"):
        compute_class_prototypes(features, labels, num_classes=2)


def test_predict_by_cosine_similarity_picks_nearest_prototype():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # class 0 = +x, class 1 = +y
    eval_features = torch.tensor(
        [
            [5.0, 0.1],  # close to +x -> class 0
            [0.1, 5.0],  # close to +y -> class 1
        ]
    )

    predictions = predict_by_cosine_similarity(eval_features, prototypes)

    assert predictions.tolist() == [0, 1]


def test_predict_output_shape_matches_number_of_evaluation_samples():
    prototypes = torch.randn(4, 6)
    prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)
    eval_features = torch.randn(10, 6)

    predictions = predict_by_cosine_similarity(eval_features, prototypes)

    assert predictions.shape == (10,)
    assert predictions.min() >= 0
    assert predictions.max() < 4
