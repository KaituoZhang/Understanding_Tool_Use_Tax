from pipeline.mechanism_ablation.gsm8k_semantic_distractor.data_loader import load_problems
from pipeline.mechanism_ablation.gsm8k_semantic_distractor.run_mechanism_ablation import run_mechanism_ablation


def test_mechanism_loader_reads_public_variants():
    problems = load_problems("data/gsm8k_semantic_distractor/sample_benchmark.jsonl")
    assert len(problems) == 10
    assert sorted({problem.variant for problem in problems}) == ["HU", "PED", "SP", "TB", "base"]


def test_mechanism_ablation_mock_backend_runs():
    results = run_mechanism_ablation(
        data_path="data/gsm8k_semantic_distractor/sample_benchmark.jsonl",
        fc_model="mock-oracle",
        fc_api_base="mock://local",
        fc_api_key=None,
        variants=["base", "TB", "PED", "HU", "SP"],
        limit=1,
        temperature=0.0,
        max_tokens=256,
        seed=42,
        save_dir=None,
        sleep=0.0,
        only=["fc_baseline", "fc_noop"],
    )
    assert set(results.keys()) == {"fc_baseline", "fc_noop"}
    assert all(results[condition] for condition in results)
