import numpy as np
import pytest

torch = pytest.importorskip("torch")

from utils.metrics import metrics, metrics_tensor, metrics_tensor_one_patience


@pytest.mark.parametrize(
    ("ground_truth", "prediction", "expected"),
    [
        ([1, 1, 0, 0], [1, 1, 0, 0], (0.0, 0.0, 1.0, 1.0)),
        ([1, 0, 0, 0], [0, 1, 0, 0], (0.3332, 0.9990, 0.0, 0.0)),
        ([0, 0, 0, 0], [0, 0, 0, 0], (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_numpy_and_tensor_metrics_agree(ground_truth, prediction, expected):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8).reshape(1, 1, 2, 2)
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(1, 1, 2, 2)

    numpy_result = metrics(ground_truth, prediction)
    tensor_result = metrics_tensor(torch.from_numpy(ground_truth), torch.from_numpy(prediction))

    assert numpy_result == pytest.approx(expected, abs=1e-3)
    assert tensor_result == pytest.approx(numpy_result)
    assert metrics_tensor_one_patience(
        torch.from_numpy(ground_truth), torch.from_numpy(prediction)
    ) == pytest.approx(tensor_result)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        metrics(np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 2, 3)))


def test_tensor_metrics_work_on_cuda_without_cpu_conversion():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    ground_truth = torch.tensor([[[[1, 0], [0, 0]]]], device="cuda")
    prediction = ground_truth.clone()
    assert metrics_tensor(ground_truth, prediction)[-1] == pytest.approx(1.0, abs=1e-3)
