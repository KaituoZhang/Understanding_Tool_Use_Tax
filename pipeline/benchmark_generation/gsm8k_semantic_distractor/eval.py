import json
import logging
import os
from typing import Any, Dict, List, Tuple

import eval.gsm8k_semantic_distractor.main as dataset_main
from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    NOISY_VARIANTS,
    dump_jsonl,
)
from pipeline.benchmark_generation.gsm8k_semantic_distractor.assemble import assemble_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.build_core import build_core_records
from pipeline.benchmark_generation.gsm8k_semantic_distractor.export_from_hf import export_gsm8k_split
from pipeline.benchmark_generation.gsm8k_semantic_distractor.generate_noise import (
    generate_noise_records,
)
from pipeline.benchmark_generation.gsm8k_semantic_distractor.parse_schema import (
    parse_schema_records,
)
from pipeline.benchmark_generation.gsm8k_semantic_distractor.validate import validate_noise_records


logger = logging.getLogger("main")


def _raw_results_dir(config: Dict[str, Any]) -> str:
    management_config = config["configs"]["management_config"]
    return os.path.join(
        management_config["output_folder_dir"],
        management_config["sub_dir"]["raw_results_folder"],
    )


def _write_stage(records: List[Dict[str, Any]], path: str) -> str:
    dump_jsonl(records, path)
    logger.info("Wrote %s records to %s", len(records), path)
    return path


def _load_or_export_raw_records(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eval_config = config["configs"]["eval_config"]
    pipeline_config = config["configs"]["pipeline_config"]
    export_config = pipeline_config.get("export", {})

    if export_config.get("source") == "huggingface":
        split_name = export_config.get("split", "train")
        export_records = export_gsm8k_split(
            config_name=export_config.get("config", "main"),
            split_name=split_name,
            limit=export_config.get("limit"),
        )
        metadata = {
            "source": "huggingface",
            "config": export_config.get("config", "main"),
            "split": split_name,
            "limit": export_config.get("limit"),
        }
        return export_records, metadata

    raw_records = dataset_main.load_raw_examples(config)
    return raw_records, {
        "source": "local_file",
        "path": eval_config["raw_data_path"],
    }


def run_generation_pipeline(config: Dict[str, Any]):
    pipeline_config = config["configs"]["pipeline_config"]
    output_dir = _raw_results_dir(config)

    raw_records, source_metadata = _load_or_export_raw_records(config)
    schema_records = parse_schema_records(raw_records)
    core_records = build_core_records(
        schema_records,
        evidence_mode=pipeline_config.get("evidence_mode", "heuristic"),
    )
    noisy_records = generate_noise_records(
        core_records,
        generator_config=pipeline_config.get("noise_generator", {}),
        noise_plan=pipeline_config.get("noise_plan", {}),
    )
    validated_records = validate_noise_records(noisy_records)
    benchmark_rows = assemble_records(
        validated_records,
        assembly_config=pipeline_config.get("assembly", {}),
    )

    stage_paths = {
        "01_schema": _write_stage(schema_records, os.path.join(output_dir, "01_schema.jsonl")),
        "02_core": _write_stage(core_records, os.path.join(output_dir, "02_core.jsonl")),
        "03_noise": _write_stage(noisy_records, os.path.join(output_dir, "03_noise.jsonl")),
        "04_validated": _write_stage(validated_records, os.path.join(output_dir, "04_validated.jsonl")),
        "05_benchmark": _write_stage(benchmark_rows, os.path.join(output_dir, "05_benchmark.jsonl")),
    }

    by_variant = {}
    for row in benchmark_rows:
        by_variant[row["variant"]] = by_variant.get(row["variant"], 0) + 1

    processed_results = {
        "source": source_metadata,
        "n_raw_examples": len(raw_records),
        "n_benchmark_rows": len(benchmark_rows),
        "variants": by_variant,
        "noisy_variants": NOISY_VARIANTS,
        "benchmark_path": stage_paths["05_benchmark"],
    }
    raw_results = {
        "stage_paths": stage_paths,
        "preview": benchmark_rows[: min(3, len(benchmark_rows))],
    }
    logger.info("Generation summary: %s", json.dumps(processed_results, indent=4))
    return raw_results, processed_results
