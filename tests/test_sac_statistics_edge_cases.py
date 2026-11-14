'''Tests for SAC statistical summary edge case handling.'''

import json
import pytest
from pathlib import Path
from adaptive_rl.experiments.sac_shift_runner import AdaptiveShiftRunner, SACAdaptiveShiftRunner

def _run_and_load(tmp_path, seeds, quick=False):
    runner = AdaptiveShiftRunner(training_seeds=seeds, output_dir=str(tmp_path / "benchmarks"), quick_test_mode=quick)
    report = runner.run()
    json_file = Path(runner.output_dir) / "adaptive_vs_fixed.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return report, data

def test_zero_replicates(tmp_path):
    # No training seeds should produce insufficient_data status
    report, data = _run_and_load(tmp_path, [], quick=True)
    stats = data["statistics"]
    assert stats["status"] == "insufficient_data"
    # All inferential values should be null
    for key in ["mean_d", "std_d", "se_d", "t_statistic", "p_value_onesided",
                "confidence_interval_95", "cohens_dz", "bootstrap_ci_95",
                "sign_test_p", "wilcoxon_p"]:
        assert stats[key] is None

def test_one_replicate_not_estimable_quick(tmp_path):
    report, data = _run_and_load(tmp_path, [12345], quick=True)
    stats = data["statistics"]
    assert stats["status"] == "not_estimable"
    # mean_d should be the single difference value
    assert isinstance(stats["mean_d"], float)
    # Other inferential fields should be null
    for key in ["std_d", "se_d", "t_statistic", "p_value_onesided",
                "confidence_interval_95", "cohens_dz", "bootstrap_ci_95",
                "sign_test_p", "wilcoxon_p"]:
        assert stats[key] is None

def test_one_replicate_not_estimable_full(tmp_path):
    report, data = _run_and_load(tmp_path, [12345], quick=False)
    stats = data["statistics"]
    assert stats["status"] == "not_estimable"

@pytest.mark.parametrize("num_seeds", [2, 3, 4, 5, 6, 7])
def test_insufficient_replicates(tmp_path, num_seeds):
    seeds = list(range(10000, 10000 + num_seeds))
    report, data = _run_and_load(tmp_path, seeds, quick=True)
    stats = data["statistics"]
    assert stats["status"] == "insufficient_replicates"
    for key in ["mean_d", "std_d", "se_d", "t_statistic", "p_value_onesided",
                "confidence_interval_95", "cohens_dz", "bootstrap_ci_95",
                "sign_test_p", "wilcoxon_p"]:
        assert stats[key] is None

def test_eight_replicates_full(tmp_path):
    seeds = list(range(20000, 20008))  # eight seeds
    report, data = _run_and_load(tmp_path, seeds, quick=True)
    stats = data["statistics"]
    assert stats["status"] == "evaluated"
    # Inferential fields should be present and numeric
    for key in ["mean_d", "std_d", "se_d", "t_statistic", "p_value_onesided",
                "confidence_interval_95", "cohens_dz", "bootstrap_ci_95",
                "sign_test_p", "wilcoxon_p"]:
        assert stats[key] is not None
