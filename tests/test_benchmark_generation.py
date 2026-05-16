from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import load_jsonl
from pipeline.benchmark_generation.gsm8k_semantic_distractor.assemble import assemble_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.build_core import build_core_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.generate_noise import generate_noise_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.parse_schema import parse_schema_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.validate import validate_noise_records


def test_mock_generation_pipeline_runs_end_to_end():
    raw_records = load_jsonl("data/gsm8k_semantic_distractor/sample_raw.jsonl")
    schema_records = parse_schema_records(raw_records)
    core_records = build_core_records(schema_records)
    noisy_records = generate_noise_records(
        core_records,
        generator_config={"provider": "mock"},
        noise_plan={"defaults": {"before": 2, "after": 2}, "per_variant": {}, "max_total_per_variant": 4},
    )
    validated_records = validate_noise_records(noisy_records)
    benchmark_rows = assemble_records(validated_records, {"variants": ["TB", "PED", "HU", "SP"]})

    assert len(benchmark_rows) == len(raw_records) * 5
    assert {row["variant"] for row in benchmark_rows} == {"base", "TB", "PED", "HU", "SP"}
    assert all("meta" in row for row in benchmark_rows)
