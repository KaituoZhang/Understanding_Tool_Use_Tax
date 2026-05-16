import logging
logger = logging.getLogger("main")

import os
import sys
import json
import argparse
import shutil

# Capture execution dir before changing to base dir
execution_dir = os.getcwd()

# Setup base dir for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(base_dir)
os.chdir(base_dir)

import utils.general_utils as general_utils
from configs import global_setting


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation pipeline')
    parser.add_argument('--exp_desc', type=str, default='experiment', help='Experiment description')

    parser.add_argument('--pipeline_config_dir', type=str, required=True, help='Path to pipeline config JSON')
    parser.add_argument('--eval_config_dir', type=str, required=True, help='Path to eval config JSON')
    parser.add_argument('--management_config_dir', type=str, required=True, help='Path to management config JSON')

    parser.add_argument('--output_folder_dir', type=str, default=None, help='Path to output folder (defaults to global_setting.default_output_dir in global_setting.py)')
    parser.add_argument('--overwrite', type=str, default='allowed', help='Overwrite behavior if output folder exists (allowed/disabled)')
    parser.add_argument('--job_post_via', type=str, default='terminal', help='Job submission method')

    args = parser.parse_args()

    # Fall back to global_setting.default_output_dir from global_setting.py if not provided
    if args.output_folder_dir is None:
        args.output_folder_dir = global_setting.default_output_dir
        logger.info(f'Using global_setting.default_output_dir from global_setting.py: {global_setting.default_output_dir}')

    # Normalize path to remove double slashes, then ensure trailing slash
    args.output_folder_dir = os.path.normpath(args.output_folder_dir) + '/'

    return args



def backup_code_files(base_dir, backup_folder_dir, inclusion_list, exclusion_list):
    """Backup code files to the output folder based on inclusion/exclusion lists.

    Args:
        base_dir: root directory of the project
        backup_folder_dir: destination folder for backups
        inclusion_list: list of paths to include (files or directories)
        exclusion_list: list of paths to exclude

    Note: if a path is in both lists, it will be included (for safety).
    """
    def should_exclude(path, inclusion_list, exclusion_list):
        """Check if a path should be excluded based on exclusion_list.

        If a path is exactly in both lists, it is included (for safety).
        """
        # If exactly in inclusion_list, never exclude (takes priority)
        if path in inclusion_list or path.rstrip('/') in [i.rstrip('/') for i in inclusion_list]:
            return False

        # Check if path matches any exclusion pattern
        for excluded in exclusion_list:
            # Match prefix (e.g., "data/" matches "data/foo.txt")
            if path.startswith(excluded):
                return True
            # Match as path component (e.g., "__pycache__/" matches "eval/__pycache__/")
            excluded_name = excluded.rstrip('/')
            if '/' + excluded_name + '/' in path or '/' + excluded_name in path:
                return True
        return False

    for item in inclusion_list:
        src_path = os.path.join(base_dir, item)
        dst_path = os.path.join(backup_folder_dir, item)

        if not os.path.exists(src_path):
            logger.warning(f'Backup source {src_path} does not exist, skipping.')
            continue

        if os.path.isfile(src_path):
            # Copy single file
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            logger.info(f'Backed up file {item}')
        elif os.path.isdir(src_path):
            # Copy directory, respecting exclusion list
            for root, dirs, files in os.walk(src_path):
                rel_root = os.path.relpath(root, base_dir)

                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not should_exclude(os.path.join(rel_root, d) + '/', inclusion_list, exclusion_list)]

                for file in files:
                    rel_file_path = os.path.join(rel_root, file)
                    if not should_exclude(rel_file_path, inclusion_list, exclusion_list):
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(backup_folder_dir, rel_file_path)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)

            logger.info(f'Backed up directory {item}')


def register_args_and_configs(args):

    # Helper functions.
    def make_folder_if_not_already_exist(folder_desc, folder_path):
        created_flag = not os.path.isdir(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f'{folder_desc} {args.output_folder_dir} {"created" if created_flag else "already exists"}.')

    def load_json_file(file_desc, file_path):
        with open(file_path) as tager_file_f:
            target_file = json.load(tager_file_f)
            logger.info(f'{file_desc} file {file_path} is loaded.')
        return target_file
    
    def dump_json_file(file_desc, file_object, file_output_path):
        with open(file_output_path, "w+") as output_file_f:
            json.dump(file_object, output_file_f, indent = 4)
            logger.info(f'{file_desc} file saved to {file_output_path}.')


    # Print exact args
    logger.info(f'Args:')
    for i, a in enumerate(sys.argv):
        logger.info(f'{i} {repr(a)}')

    # Load input configs.
    pipeline_config = load_json_file('Input pipeline config', args.pipeline_config_dir)
    eval_config = load_json_file('Input eval config', args.eval_config_dir)
    management_config = load_json_file('Input management config', args.management_config_dir)


    # Check if output folder exists and handle overwrite logic
    if os.path.isdir(args.output_folder_dir):
        folder_contents = os.listdir(args.output_folder_dir)
        if folder_contents:
            if args.overwrite == 'disabled':
                raise FileExistsError(f'Output folder {args.output_folder_dir} already exists with contents. Use --overwrite allowed to overwrite.')
            else:
                logger.warning(f'Output folder {args.output_folder_dir} already exists with contents. Overwriting.')
        else:
            logger.info(f'Output folder {args.output_folder_dir} already exists but is empty.')

    # Make output folder and its subdirs.
    make_folder_if_not_already_exist('Output folder', args.output_folder_dir)
    make_folder_if_not_already_exist('Input configs folder', os.path.join(args.output_folder_dir, management_config['sub_dir']['input_configs_folder']))
    make_folder_if_not_already_exist('Backup folder', os.path.join(args.output_folder_dir, management_config['sub_dir']['backup_folder']))
    make_folder_if_not_already_exist('Raw results folder', os.path.join(args.output_folder_dir, management_config['sub_dir']['raw_results_folder']))

    # Backup code files based on backup_scope config.
    backup_folder_dir = os.path.join(args.output_folder_dir, management_config['sub_dir']['backup_folder'])
    backup_scope = management_config['backup_scope']
    backup_code_files(base_dir, backup_folder_dir, backup_scope['inclusion_list'], backup_scope['exclusion_list'])

    # Copy input configs to output dir.
    input_pipeline_config_path = os.path.join(args.output_folder_dir, management_config['sub_dir']['input_configs_folder'], 'input_pipeline_config.json')
    dump_json_file('Input pipeline config', pipeline_config, input_pipeline_config_path)
    input_eval_config_path = os.path.join(args.output_folder_dir, management_config['sub_dir']['input_configs_folder'], 'input_eval_config.json')
    dump_json_file('Input eval config', eval_config, input_eval_config_path)
    input_management_config_path = os.path.join(args.output_folder_dir, management_config['sub_dir']['input_configs_folder'], 'input_management_config.json')
    dump_json_file('Input management config', management_config, input_management_config_path)


    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.
    config = dict()
    config['configs'] = dict()

    config['configs']['pipeline_config'] = pipeline_config
    config['configs']['eval_config'] = eval_config
    config['configs']['management_config'] = management_config

    # Register global settings for reproducibility
    config['configs']['management_config']['global_setting'] = {
        'SEED': global_setting.SEED,
        'timezone': global_setting.timezone,
        'dataset_dir': global_setting.dataset_dir,
        'model_checkpoint_dir': global_setting.model_checkpoint_dir,
        'default_output_dir': global_setting.default_output_dir,
        'hf_access_token': global_setting.hf_access_token
    }
    config['configs']['management_config']['execution_dir'] = execution_dir
    config['configs']['management_config']['args'] = sys.argv
    config['configs']['management_config']['exp_desc'] = args.exp_desc
    # Store paths as absolute paths for reproducibility
    config['configs']['management_config']['pipeline_config_dir'] = os.path.abspath(args.pipeline_config_dir)
    config['configs']['management_config']['eval_config_dir'] = os.path.abspath(args.eval_config_dir)
    config['configs']['management_config']['management_config_dir'] = os.path.abspath(args.management_config_dir)
    config['configs']['management_config']['output_folder_dir'] = os.path.abspath(args.output_folder_dir)

    config['configs']['management_config']['overwrite'] = args.overwrite
    config['configs']['management_config']['job_post_via'] = args.job_post_via
    if config['configs']['management_config']['job_post_via'] == 'slurm_sbatch':     # Add slurm info to config['management'] if the job is triggered via slurm sbatch.
        try:
            config['configs']['management_config']['slurm_info'] = general_utils.general_utils.register_slurm_sbatch_info()
        except Exception:
             config['configs']['management_config']['job_post_via'] == 'terminal'      # Likely not a slurm job, rollback to terminal post.

    return config


def format_inputs_for_chat_template(batched_input, tokenizer, chat_template, add_generation_prompt, enable_thinking):
    """Apply chat template to a batch of input strings.

    Args:
        batched_input: list of input strings
        tokenizer: HuggingFace tokenizer with chat template
        chat_template: template type ("default" uses tokenizer's built-in template)
        add_generation_prompt: whether to add assistant turn prompt at the end
        enable_thinking: whether to enable thinking mode (Qwen3-specific)

    Returns:
        list of formatted strings with chat template applied
    """
    formatted = []
    for text in batched_input:
        messages = [{"role": "user", "content": text}]
        if chat_template == "default":
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking))
        else:
            raise ValueError(f"Unknown chat_template: {chat_template}")
    return formatted