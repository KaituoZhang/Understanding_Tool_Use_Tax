import logging
from typing import Any, Dict, List

from .gsm8k_semantic_distractor_utils import load_jsonl


logger = logging.getLogger("main")


def load_raw_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_path = config["configs"]["eval_config"]["raw_data_path"]
    records = load_jsonl(data_path)
    logger.info("Loaded %s raw examples from %s", len(records), data_path)
    return records


def load_benchmark_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_path = config["configs"]["eval_config"]["benchmark_data_path"]
    records = load_jsonl(data_path)
    logger.info("Loaded %s benchmark rows from %s", len(records), data_path)
    return records
