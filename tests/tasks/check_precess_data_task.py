# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""check preprocess data"""

from tasks.base_task import BaseTask, TaskResut
from tools.color_logger import create_color_logger
import os, time
from copy import deepcopy
import json, re, yaml
import subprocess
import os
import sys
import shutil
import random
from typing import Dict, List, Any

logger = create_color_logger(name=__name__)
import json
import yaml
import glob
from datetime import timedelta

class PrecessDataCheckTask(BaseTask):
    """PrecessDataCheckTask"""
    def __init__(self,
                 model_description: Dict[str, Any],
                 task_description: Dict[str, Any],
                 model: List[Dict[Any, Any]],
                 model_configer: object,
                 args
                ) -> None:
        super().__init__(model_description,
                 task_description,
                 model,
                 model_configer,
                 args,
                 task_type = "preprocess_data",
                )
        self.class_name = self.__class__.__name__

    
    def deal_output(self, model_config, step_stage):
        """
        Handle output directory for data preprocessing, ensure directory is empty or recreated
        Args:
            model_config: Model configuration information
            step_stage: Preprocessing stage (llm_pretrain, llm_sft, vlm, offline_packing)
        """
        if step_stage == "llm_pretrain":
            output_prefix = model_config["output_prefix"]
            if os.path.isdir(os.path.dirname(output_prefix)):
                shutil.rmtree(os.path.dirname(output_prefix))
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        elif step_stage == "llm_sft":
            output_prefix = model_config["output_prefix"]
            if os.path.isdir(output_prefix):
                shutil.rmtree(output_prefix)
            os.makedirs(output_prefix, exist_ok=True)
        elif step_stage == "vlm":
            output_prefix = model_config["output_prefix"]
            if os.path.isdir(output_prefix):
                shutil.rmtree(output_prefix)
            os.makedirs(output_prefix, exist_ok=True)
        elif step_stage == "offline_packing":
            packing_config_path = model_config.get("packing_config_path")
            if packing_config_path and os.path.exists(packing_config_path):
                with open(packing_config_path, 'r', encoding='utf-8') as f:
                    packing_config = yaml.safe_load(f)
                packed_wds_dir = packing_config.get('data', {}).get('packed_wds_dir')
                if packed_wds_dir and os.path.isdir(packed_wds_dir):
                    shutil.rmtree(packed_wds_dir)
                if packed_wds_dir:
                    os.makedirs(packed_wds_dir, exist_ok=True)
        else:
            logger.error(f"deal_output does not support {step_stage} mode !!!")
            sys.exit(1)

    def assert_preprocess_data(self, model_config, step_stage):
        """
        Verify if data preprocessing results meet expectations
        Args:
            model_config: Model configuration information
            step_stage: Preprocessing stage (llm_pretrain, llm_sft, vlm, offline_packing)
        """
        if step_stage == "llm_pretrain":
            output_prefix = model_config["output_prefix"]
            output_dir = os.path.dirname(output_prefix)

            # Search for .bin and .idx files under path
            bin_files = glob.glob(os.path.join(output_dir, '*.bin'))
            idx_files = glob.glob(os.path.join(output_dir, '*.idx'))

            # Use assertion to ensure .bin and .idx files are found
            assert len(bin_files) > 0, "No .bin files found in " + output_dir
            assert len(idx_files) > 0, "No .idx files found in " + output_dir
            logger.info(f"LLM pretrain preprocess data check passed: found {len(bin_files)} .bin files and {len(idx_files)} .idx files in {output_dir}")

        elif step_stage == "llm_sft":
            output_dir = model_config["output_prefix"]
            train_dir = os.path.join(output_dir, 'train')
            dataset_info_files = glob.glob(os.path.join(train_dir, 'dataset_info.json'))
            state_files = glob.glob(os.path.join(train_dir, 'state.json'))
            assert len(dataset_info_files) > 0, "No dataset_info files found in " + train_dir
            assert len(state_files) > 0, "No state files found in " + train_dir
            logger.info(f"LLM SFT preprocess data check passed: found dataset_info.json and state.json in {train_dir}")
        
        elif step_stage == "vlm":
            output_dir = model_config["output_prefix"]
            info_yaml_path = os.path.join(output_dir, '.nv-meta', '.info.yaml')
            assert os.path.exists(info_yaml_path), f".nv-meta/.info.yaml not found in {output_dir}"

            with open(info_yaml_path, 'r', encoding='utf-8') as f:
                info_data = yaml.safe_load(f)
            shard_counts = info_data.get('shard_counts', {})
            assert len(shard_counts) > 0, f"No shard_counts found in {info_yaml_path}"
            
            total_shard_count = sum(shard_counts.values())
            train_json_file = model_config.get("train_json_file_name")        
            with open(train_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            expected_count = len(json_data)  
            assert total_shard_count == expected_count, \
                f"Shard count mismatch: .info.yaml has {total_shard_count} samples, but JSON has {expected_count} messages"
            logger.info(f"VLM preprocess data check passed: {total_shard_count} samples match {expected_count} messages")
        
        elif step_stage == "offline_packing":
            packing_config_path = model_config.get("packing_config_path")
            assert packing_config_path and os.path.exists(packing_config_path), \
                f"Packing config file not found: {packing_config_path}"
            
            with open(packing_config_path, 'r', encoding='utf-8') as f:
                packing_config = yaml.safe_load(f)
            
            packed_wds_dir = model_config.get("packed_wds_dir") or packing_config.get('data', {}).get('packed_wds_dir')
            assert packed_wds_dir and os.path.exists(packed_wds_dir), \
                f"Packed wds_dir not found: {packed_wds_dir}"
            
            # Check if .nv-meta directory exists
            nv_meta_dir = os.path.join(packed_wds_dir, '.nv-meta')
            assert os.path.exists(nv_meta_dir), f".nv-meta directory not found in {packed_wds_dir}"

            packed_info_yaml_path = os.path.join(nv_meta_dir, '.info.yaml')
            assert os.path.exists(packed_info_yaml_path), \
                f"Packed .nv-meta/.info.yaml not found in {packed_wds_dir}"
            
            with open(packed_info_yaml_path, 'r', encoding='utf-8') as f:
                packed_info_data = yaml.safe_load(f)
            packed_shard_counts = packed_info_data.get('shard_counts', {})
            packed_total_count = sum(packed_shard_counts.values())
            
            # Check total count is not 0
            assert packed_total_count > 0, \
                f"Packed wds is empty: shard_counts sum is 0 in {packed_info_yaml_path}"
            
            logger.info(f"Offline packing check passed: packed {packed_total_count} samples (nonzero)")
        else:
            logger.error(f"assert_preprocess_data does not support other {step_stage} mode verification !!!")
            sys.exit(1)


    def start_loongforge_preprocess_data(self, index, step_stage, scenario_name):
        """
        Start data preprocessing task
        Args:
            index: Scenario index
            step_stage: Preprocessing stage name (e.g. llm_pretrain, llm_sft)
            scenario_name: Scenario name
        """
        step_name = "loongforge_preprocess_data"
        logger.info(f"{step_stage} {step_name} Start Running ...")

        model_config = self.__init_model_scenarios_data__(index, scenario_name, step_stage)

        # Data preprocessing
        model_name = self.model_name
        node_nums = self.input_cmd_args.node_nums
        timeout = self.input_cmd_args.timeout
        scripts_root_path = model_config["scripts_root_path"]
        model_lock_file_path = model_config["model_lock_file_path"]
        training_log_path = model_config["training_log_path"]

        # Convert config file to env variables passed to running script
        env_vars_str = self.__convert_model_config_to_env__(model_config)
        self.deal_output(model_config, step_stage)

        step_stage_path = f'{model_lock_file_path}/{step_stage}/{self.master_addr}'
        model_lock_file = f'{step_stage_path}/{self.rank_name}_lock.txt'

        script_path = f"{scripts_root_path}/executor/{step_name}/run.sh"
        new_script_path = f"{training_log_path}/precess_data_{model_name}_{self.rank_name}_run.sh"
        start_command = f"{env_vars_str} bash {script_path}"
        self.create_shell_file(model_config, script_path, new_script_path)

        # Open a new file to write script output
        training_log_file = f"{training_log_path}/precess_data#{model_name}#nodes_{self.input_cmd_args.node_nums}#{self.rank_name}#run.log"

        start_command = f"{env_vars_str} bash -c \"set -o pipefail; bash {scripts_root_path}/executor/{step_name}/run.sh |tee {training_log_file}\""
        logger.info(f"{step_stage} {step_name} Start: {start_command} .")
        if os.system(start_command) != 0:
           raise RuntimeError(f"Start {step_stage} {step_name} error, cmd is {start_command}")
        
        def _assert_and_record_preprocess():
            category = self._get_diff_category(self.model)
            case_name = f"preprocess_data:{step_stage}"
            try:
                self.assert_preprocess_data(model_config=model_config, step_stage=step_stage)
                self._record_case_result(
                    model_name,
                    case_name,
                    category,
                    True,
                    [],
                    task_name=getattr(self, "task_description", ""),
                )
                return True
            except Exception as exc:
                error_message = str(exc)
                logger.error(f"Preprocess data check failed: {model_name} {case_name}, error: {error_message}")
                self._record_case_result(
                    model_name,
                    case_name,
                    category,
                    False,
                    [step_stage],
                    error_message=error_message,
                    task_name=getattr(self, "task_description", ""),
                )
                return False

        # Wait for all pods to complete
        self.wait_async_pod_complete(
            model_lock_file,
            model_name,
            f"{scenario_name}_{step_name}",
            is_function=True,
            function=_assert_and_record_preprocess,
            raise_on_error=False,
        )

        logger.info(f"{step_stage} End {step_name}")

    def __call__(self) -> TaskResut:
        if not self.MODEL_RUNNABLE:
            logger.warn(f"{self.class_name} current model {self.model_name} does not support {self.class_name} task, skipping!!!")
            return TaskResut()

        for index, scenario in enumerate(self.model["scenarios"]):
            for scenario_name, scenario_data in scenario.items():
                if scenario_name != "preprocess_data":
                    continue
                model_name = self.model_name
                logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] Execution Start ...")
                
                for key, value in self.model["scenarios"][index][scenario_name].items():

                    # pretrain, sft:
                    step_name = key
                    logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] - [{step_name}] Execution Start ...")

                    runnable_flag = self.model["scenarios"][index][scenario_name][step_name].get("RUNNABLE_FLAG")
                    if runnable_flag is None or (isinstance(runnable_flag, str) and runnable_flag.lower() == "true") or runnable_flag is True:
                        self.start_loongforge_preprocess_data(index, step_name, scenario_name)
                        step_scenario_lock_file = os.path.join(self.model["model_lock_file_path"], scenario_name, step_name, self.master_addr, f"{self.rank_name}_lock.txt")
                        self.wait_async_pod_complete(step_scenario_lock_file, model_name, f"{scenario_name}_{step_name}")

                        logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] - [{step_name}] Completed \n")
                    else:
                        logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] - [{step_name}] RUNNABLE_FLAG is false, skipping.\n")

                logger.info(f"{self.class_name} Model [{model_name}] - [{scenario_name}] Execution End \n")
    
        return TaskResut()