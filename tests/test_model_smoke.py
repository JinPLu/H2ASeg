"""Small real-model smoke tests for the documented runtime environment."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from monai.inferers import sliding_window_inference

from inference import InferencePredictor
from model.H2ASeg import H2ASeg
from train import create_grad_scaler
from utils.model_config import H2ASegModelConfig


SMOKE_CONFIG = H2ASegModelConfig(
    patch_size=(64, 64, 32),
    num_channels=(4, 4, 4, 8, 16),
    attention_dropout=0.0,
    encoder_dropout=0.0,
    decoder_dropout=0.0,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device_name", DEVICES)
def test_real_h2aseg_forward_backward_and_sliding_window(device_name: str) -> None:
    device = torch.device(device_name)
    use_amp = device.type == "cuda"
    model = H2ASeg(**SMOKE_CONFIG.to_model_kwargs()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scaler = create_grad_scaler(use_amp)
    model.train()
    ct = torch.randn(
        (1, 1, *SMOKE_CONFIG.patch_size), device=device, requires_grad=True
    )
    pet = torch.randn_like(ct, requires_grad=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        outputs = model(ct, pet)
        loss = sum(output.square().mean() for output in outputs)
    assert len(outputs) == 5
    assert all(output.shape == (1, 2, *SMOKE_CONFIG.patch_size) for output in outputs)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert ct.grad is not None
    assert pet.grad is not None

    model.eval()
    predictor = InferencePredictor(model)
    volume = torch.randn((1, 2, 72, 64, 32), device=device)
    with torch.inference_mode():
        prediction = sliding_window_inference(
            volume,
            roi_size=SMOKE_CONFIG.patch_size,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.25,
        )
    assert prediction.shape == (1, 2, 72, 64, 32)
