import json
import logging
import os
from typing import Any, Dict

from pipeline.mechanism_ablation.gsm8k_semantic_distractor.run_mechanism_ablation import (
    _build_summary,
    run_mechanism_ablation,
)


logger = logging.getLogger("main")


def run_mechanism_pipeline(config: Dict[str, Any]):
    pipeline_config = config["configs"]["pipeline_config"]
    eval_config = config["configs"]["eval_config"]
    management_config = config["configs"]["management_config"]
    save_dir = os.path.join(
        management_config["output_folder_dir"],
        management_config["sub_dir"]["raw_results_folder"],
        "mechanism_ablation",
    )

    results = run_mechanism_ablation(
        data_path=pipeline_config.get("data_path", eval_config["benchmark_data_path"]),
        fc_model=pipeline_config.get("model", "mock-oracle"),
        fc_api_base=pipeline_config.get("api_base", "mock://local"),
        fc_api_key=pipeline_config.get("api_key"),
        variants=pipeline_config.get("variants", eval_config.get("variants")),
        limit=pipeline_config.get("limit"),
        offset=int(pipeline_config.get("offset", 0)),
        temperature=float(pipeline_config.get("temperature", 0.0)),
        max_tokens=int(pipeline_config.get("max_tokens", 1024)),
        seed=int(pipeline_config.get("seed", 42)),
        save_dir=save_dir,
        sleep=float(pipeline_config.get("sleep", 0.0)),
        only=pipeline_config.get("only"),
        g_step_model_path=pipeline_config.get("g_step_model_path"),
        g_commit_model_path=pipeline_config.get("g_commit_model_path"),
        g_step_threshold=float(pipeline_config.get("g_step_threshold", 0.05)),
        g_commit_threshold=float(pipeline_config.get("g_commit_threshold", 0.5)),
        g_step_max_extra_turns=int(pipeline_config.get("g_step_max_extra_turns", 3)),
        include_extended_gate_conditions=bool(
            pipeline_config.get("include_extended_gate_conditions", False)
        ),
    )

    summary = _build_summary(results)
    raw_results = {
        "save_dir": save_dir,
        "conditions": list(results.keys()),
        "summary_path": os.path.join(save_dir, "mechanism_report.json"),
    }
    processed_results = {
        "data_path": pipeline_config.get("data_path", eval_config["benchmark_data_path"]),
        "model": pipeline_config.get("model", "mock-oracle"),
        "conditions": list(summary.keys()),
        "summary": summary,
    }
    logger.info("Mechanism summary: %s", json.dumps(processed_results, indent=4))
    return raw_results, processed_results
