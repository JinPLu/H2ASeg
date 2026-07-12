"""Small real-model smoke tests for the documented runtime environment."""

from __future__ import annotations

import warnings

import pytest


torch = pytest.importorskip("torch")

from model.Attention import MCSA
from model.H2ASeg import H2ASeg
from model.MCSA import ResUNet_MCSA_Encoder
from model.modules import IRC1, IRC3
from utils.model_config import H2ASegModelConfig


SMOKE_CONFIG = H2ASegModelConfig(
    patch_size=(64, 64, 32),
    num_channels=(4, 4, 4, 8, 16),
    attention_dropout=0.0,
    encoder_dropout=0.0,
    decoder_dropout=0.0,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("block_type", [IRC1, IRC3])
def test_irc_normalizes_input_channels_without_runtime_warning(block_type) -> None:
    block = block_type(feat_in=4, feat_out=2).eval()
    assert block.conv[0].num_features == 4

    input_tensor = torch.randn((1, 4, 4, 4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        output = block(input_tensor)

    assert output.shape == (1, 2, 4, 4, 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 2, "n_classes": 2}, "in_channels must be the integer 1"),
        ({"in_channels": 1, "n_classes": 3}, "n_classes must be the integer 2"),
        ({"in_channels": True, "n_classes": 2}, "in_channels must be the integer 1"),
        ({"in_channels": 1, "n_classes": True}, "n_classes must be the integer 2"),
        ({"in_channels": 1.0, "n_classes": 2}, "in_channels must be the integer 1"),
        ({"in_channels": 1, "n_classes": 2.0}, "n_classes must be the integer 2"),
    ],
)
def test_h2aseg_direct_constructor_rejects_unsupported_io_contract(
    kwargs: dict[str, int | float | bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        H2ASeg(
            patch_size=(32, 32, 16),
            num_channels=(4, 4, 4, 8, 16),
            window_size=(4, 4, 2),
            **kwargs,
        )


def test_mcsa_default_window_preserves_legacy_merge_bias() -> None:
    without_bias = MCSA(in_dim=4, dropout=0.0, conv_bias=False).eval()
    with_bias = MCSA(in_dim=4, dropout=0.0, conv_bias=True).eval()

    assert without_bias.window_size == (8, 8, 4)
    assert without_bias.pool_size == 4
    # win_merge historically always had bias. Preserve those keys so old
    # conv_bias=False checkpoints remain strict-load compatible.
    assert without_bias.win_merge_ct.bias is not None
    assert without_bias.win_merge_pet.bias is not None
    assert with_bias.win_merge_ct.bias is not None
    assert with_bias.win_merge_pet.bias is not None

    ct = torch.randn((1, 4, 8, 8, 4))
    pet = torch.randn_like(ct)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with torch.inference_mode():
            ct_output, pet_output = without_bias(ct, pet)

    assert ct_output.shape == ct.shape
    assert pet_output.shape == pet.shape


@pytest.mark.parametrize("window_size", [(8, 8), (8, 8, 0), (8, 8, 1.5)])
def test_mcsa_rejects_invalid_window_size(window_size) -> None:
    with pytest.raises(ValueError, match="three positive integers"):
        MCSA(in_dim=4, window_size=window_size)


def test_mcsa_encoder_accepts_default_and_preserves_five_stage_window_mapping() -> None:
    default_encoder = ResUNet_MCSA_Encoder(
        in_channels=1,
        num_channels=(4, 4, 4, 8, 16),
    )
    staged_windows = torch.tensor(
        [
            (2, 2, 1),
            (8, 8, 4),
            (8, 8, 4),
            (4, 4, 2),
            (2, 2, 1),
        ]
    )
    staged_encoder = ResUNet_MCSA_Encoder(
        in_channels=1,
        num_channels=(4, 4, 4, 8, 16),
        window_size=staged_windows,
    )

    assert default_encoder.sa2.window_size == (8, 8, 4)
    assert default_encoder.sa3.window_size == (8, 8, 4)
    assert staged_encoder.sa2.window_size == (8, 8, 4)
    assert staged_encoder.sa3.window_size == (8, 8, 4)
    assert {
        name: tuple(value.shape)
        for name, value in default_encoder.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in staged_encoder.state_dict().items()
    }


@pytest.mark.parametrize("device_name", DEVICES)
def test_real_h2aseg_forward_backward_and_sliding_window(device_name: str) -> None:
    pytest.importorskip("monai")
    from monai.inferers import sliding_window_inference

    from inference import InferencePredictor
    from train import create_grad_scaler

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
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
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
