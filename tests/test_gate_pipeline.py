from pathlib import Path

from pipeline.mechanism_ablation.gsm8k_semantic_distractor.gate.gate_models import GStepGate
from pipeline.mechanism_ablation.gsm8k_semantic_distractor.gate.train_gate import (
    train_gstep,
)
from pipeline.mechanism_ablation.gsm8k_semantic_distractor.run_mechanism_ablation import (
    run_mechanism_ablation,
)


SAMPLE_RESULTS_DIR = Path("data/gsm8k_semantic_distractor/sample_gate_results")


def test_gate_training_writes_artifacts(tmpdir):
    out_dir = Path(str(tmpdir)) / "gate_artifacts"

    g_step_summary = train_gstep(SAMPLE_RESULTS_DIR, out_dir)

    assert (out_dir / "g_step.pkl").exists()
    assert g_step_summary["n_positive"] > 0
    assert g_step_summary["feature_dim"] == 120
    assert g_step_summary["label_mode"] == "paper"

    g_step = GStepGate.from_artifact(str(out_dir / "g_step.pkl"))

    step_decision = g_step.decide(
        tool_trace=[{"tool": "calculator", "input": ["9*4"], "output": "36"}],
        candidate_text="I should subtract 6 before I commit.",
        pred_answer="36",
        n_chunks_seen=6,
        max_tool_calls_limit=5,
        extra_turns_used=0,
    )

    assert step_decision.action in {"continue", "commit"}


def test_gate_conditions_run_with_mock_backend(tmpdir):
    artifact_dir = Path(str(tmpdir)) / "trained"
    train_gstep(SAMPLE_RESULTS_DIR, artifact_dir)

    results = run_mechanism_ablation(
        data_path="data/gsm8k_semantic_distractor/sample_benchmark.jsonl",
        fc_model="mock-oracle",
        fc_api_base="mock://local",
        fc_api_key=None,
        variants=["base", "TB", "PED", "HU", "SP"],
        limit=1,
        offset=0,
        temperature=0.0,
        max_tokens=256,
        seed=42,
        save_dir=None,
        sleep=0.0,
        only=["gate_step", "gate_step_critic"],
        g_step_model_path=str(artifact_dir / "g_step.pkl"),
        g_step_threshold=0.05,
        g_step_max_extra_turns=3,
    )

    assert set(results.keys()) == {"gate_step", "gate_step_critic"}
    assert all(results[condition] for condition in results)
    for rows in results.values():
        assert "gate_action" in rows[0]
        assert "gate_p_continue" in rows[0]
