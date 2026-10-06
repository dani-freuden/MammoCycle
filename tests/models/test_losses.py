import torch

from src.models.model import MammoCycleModule


def test_alignment_loss_is_zero_for_identical_params():
    model = MammoCycleModule()

    params = torch.tensor([
        [0.2, -0.3, 0.1],
        [-0.4, 0.1, -0.2],
    ])

    loss = model.alignment_loss(
        reference_params=params,
        predicted_params=params,
    )

    torch.testing.assert_close(
        loss,
        torch.tensor(0.0),
        atol=1e-7,
        rtol=0,
    )
    
def test_alignment_loss_is_positive_for_rotation():
    model = MammoCycleModule()

    reference = torch.tensor([
        [0.0, 0.0, 0.0],
    ])

    predicted = torch.tensor([
        [0.0, 0.0, 0.2],
    ])

    loss = model.alignment_loss(
        reference_params=reference,
        predicted_params=predicted,
    )

    assert loss > 0