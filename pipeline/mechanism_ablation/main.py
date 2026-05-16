import datetime
import json
import os
import sys
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)

from configs.global_setting import SEED, timezone
import pipeline.pipeline_utils as pipeline_utils
import utils.config_utils as config_utils
import utils.general_utils as general_utils
import utils.logger_utils as logger_utils


general_utils.lock_seed(SEED)


def main() -> None:
    start_time = datetime.datetime.now(ZoneInfo(timezone)) if ZoneInfo is not None else datetime.datetime.now()
    args = pipeline_utils.parse_args()
    logger = logger_utils.set_logger(args.output_folder_dir, args)
    config = pipeline_utils.register_args_and_configs(args)

    logger.info(
        "Experiment %s (SEED=%s) started at %s",
        config["configs"]["management_config"]["exp_desc"],
        SEED,
        start_time,
    )
    logger.info("Config: %s", json.dumps(config, indent=4))

    dataset = config["configs"]["eval_config"]["dataset"]
    if dataset != "gsm8k_semantic_distractor":
        raise ValueError(f"Unsupported dataset for mechanism_ablation: {dataset}")

    from pipeline.mechanism_ablation.gsm8k_semantic_distractor.eval import (
        run_mechanism_pipeline,
    )

    raw_results, processed_results = run_mechanism_pipeline(config)
    config_utils.register_raw_and_processed_results(raw_results, processed_results, config)

    end_time = datetime.datetime.now(ZoneInfo(timezone)) if ZoneInfo is not None else datetime.datetime.now()
    config_utils.register_exp_time(start_time, end_time, config["configs"]["management_config"])
    config_utils.register_output_file(config)
    logger.info(
        "Experiment ended at %s. Duration: %s",
        end_time,
        config["configs"]["management_config"]["exp_duration"],
    )


if __name__ == "__main__":
    main()
