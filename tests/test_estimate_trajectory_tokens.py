from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from estimate_trajectory_tokens import calibration_ratios, estimate_token_metadata  # noqa: E402


def test_estimate_uses_empirical_qwen35_quantiles():
    calibration = calibration_ratios(
        [
            {"characters": 100, "input_tokens": 40, "supervised_ratio": 0.10},
            {"characters": 100, "input_tokens": 50, "supervised_ratio": 0.20},
            {"characters": 100, "input_tokens": 70, "supervised_ratio": 0.30},
        ]
    )
    result = estimate_token_metadata(200, calibration)

    assert result["input_tokens"] == 100
    assert result["supervised_tokens"] == 20
    assert result["input_tokens_estimate_low"] == 80
    assert result["input_tokens_estimate_high"] == 140
    assert result["token_method"] == "estimated_from_qwen35_calibration"
