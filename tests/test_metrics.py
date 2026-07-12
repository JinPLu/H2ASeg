import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from utils.metrics import metrics, metrics_tensor, metrics_tensor_one_patience


@pytest.mark.parametrize(
    ("ground_truth", "prediction", "expected"),
    [
        ([1, 1, 0, 0], [1, 1, 0, 0], (0.0, 0.0, 1.0, 1.0)),
        ([1, 0, 0, 0], [0, 1, 0, 0], (0.3332, 0.9990, 0.0, 0.0)),
    ],
)
def test_numpy_metrics_regression(ground_truth, prediction, expected):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8).reshape(1, 1, 2, 2)
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(1, 1, 2, 2)

    numpy_result = metrics(ground_truth, prediction)
    assert numpy_result == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    ("ground_truth", "prediction", "expected_overlap"),
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], (1.0, 1.0)),
        ([0, 0, 0, 0], [1, 0, 0, 0], (0.0, 0.0)),
        ([1, 0, 0, 0], [0, 0, 0, 0], (0.0, 0.0)),
        ([1, 0, 0, 0], [1, 0, 0, 0], (0.9990, 0.9995)),
    ],
)
def test_empty_mask_policy_covers_all_combinations(
    ground_truth, prediction, expected_overlap
):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8).reshape(1, 1, 2, 2)
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(1, 1, 2, 2)

    assert metrics(ground_truth, prediction)[-2:] == pytest.approx(
        expected_overlap, abs=1e-4
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 0, 0, 0]),
        ([1, 0, 0, 0], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, 0]),
    ],
)
def test_numpy_and_tensor_metrics_agree(ground_truth, prediction):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8).reshape(1, 1, 2, 2)
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(1, 1, 2, 2)

    numpy_result = metrics(ground_truth, prediction)
    tensor_result = metrics_tensor(torch.from_numpy(ground_truth), torch.from_numpy(prediction))

    assert tensor_result == pytest.approx(numpy_result)
    assert metrics_tensor_one_patience(
        torch.from_numpy(ground_truth), torch.from_numpy(prediction)
    ) == pytest.approx(tensor_result)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        metrics(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 2, 3)))


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((1, 2)), "batched channel-first"),
        (np.zeros((1, 3, 2, 2)), "one label channel or two one-hot channels"),
        (np.zeros((1, 1, 0, 2)), "zero-sized"),
        (np.asarray([[[[np.nan]]]]), "NaN or infinite"),
        (np.asarray([[[[np.inf]]]]), "NaN or infinite"),
    ],
)
def test_metrics_reject_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        metrics(mask, mask)


def test_two_channel_masks_select_foreground_channel():
    ground_truth = np.asarray([[[[0, 1]], [[1, 0]]]])
    prediction = np.asarray([[[[1, 0]]]])
    assert metrics(ground_truth, prediction)[-2:] == pytest.approx(
        (1 / (1 + 1e-3), 2 / (2 + 1e-3))
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_tensor_metrics_reject_nonfinite_masks():
    mask = torch.tensor([[[[float("nan")]]]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        metrics_tensor(mask, mask)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_tensor_metrics_work_on_cuda_without_cpu_conversion():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    ground_truth = torch.tensor([[[[1, 0], [0, 0]]]], device="cuda")
    prediction = ground_truth.clone()
    assert metrics_tensor(ground_truth, prediction)[-1] == pytest.approx(1.0, abs=1e-3)
