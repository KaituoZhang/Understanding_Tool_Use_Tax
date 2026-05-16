import logging
logger = logging.getLogger("main")

import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from configs.global_setting import SEED


def lock_seed(seed=SEED):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed



def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}
