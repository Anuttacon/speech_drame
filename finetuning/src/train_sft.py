# Copyright 2025 Jiatong Shi (Anuttacon)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import yaml
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import SFTConfig

from trainer.sft_trainer import SFTTrainer
from dataset.dataset import AudioDataset
from utils.peft_utils import create_peft_config


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default=None, metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default=None, metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default=None, metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="false", metadata={"help": "whether use wandb to report logs"})
    yaml_config: Optional[str] = field(default="conf/sft_config.yaml", metadata={"help": "path to YAML config file"})
    enable_tensorboard: Optional[bool] = field(default=True, metadata={"help": "whether to enable TensorBoard logging"})
    tensorboard_log_dir: Optional[str] = field(default=None, metadata={"help": "custom directory for TensorBoard logs"})


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sft_config_from_yaml(config: dict, data_args) -> SFTConfig:
    """Create SFTConfig from YAML configuration."""
    training_config = config.get("training", {})
    
    # Override with command line arguments if provided
    if data_args.model_name_or_path:
        training_config["model_name_or_path"] = data_args.model_name_or_path
    if data_args.out_dir:
        training_config["output_dir"] = data_args.out_dir
    if data_args.data_file:
        training_config["data_file"] = data_args.data_file
    if data_args.config_path:
        training_config["deepspeed_config"] = data_args.config_path
    
    # Handle wandb reporting
    if data_args.use_wandb == "true":
        training_config["report_to"] = ["wandb"]
    else:
        training_config["report_to"] = training_config.get("report_to", [])
    
    # Handle TensorBoard settings
    if data_args.enable_tensorboard is not None:
        training_config["enable_tensorboard"] = data_args.enable_tensorboard
    if data_args.tensorboard_log_dir is not None:
        training_config["tensorboard_log_dir"] = data_args.tensorboard_log_dir
    
    return SFTConfig(
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        output_dir=training_config.get("output_dir", "exp/sft_small_batch"),
        deepspeed=training_config.get("deepspeed_config"),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        logging_steps=training_config.get("logging_steps", 1),
        bf16=training_config.get("bf16", True),
        report_to=training_config.get("report_to", []),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        num_train_epochs=training_config.get("num_train_epochs", 2),
        max_steps=training_config.get("max_steps", 1000),
        run_name=training_config.get("run_name", "ROLE-SFT-Optimized"),
        save_steps=training_config.get("save_steps", 100),
        save_only_model=training_config.get("save_only_model", True),
        learning_rate=training_config.get("learning_rate", 5e-5),
        warmup_steps=training_config.get("warmup_steps", 100),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        weight_decay=training_config.get("weight_decay", 0.01),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        optim=training_config.get("optim", "adamw_torch"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        group_by_length=training_config.get("group_by_length", False),
        length_column_name=training_config.get("length_column_name", "length"),
        ignore_data_skip=training_config.get("ignore_data_skip", False),
        dataloader_drop_last=training_config.get("dataloader_drop_last", True),
    )


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    # Load YAML configuration
    config = load_yaml_config(data_args.yaml_config)
    
    # Create training dataset
    data_file = data_args.data_file or config.get("training", {}).get("data_file", "data/train_evalv1-v2/data.jsonl")
    train_dataset = AudioDataset(data_file)

    # Create training arguments from YAML
    training_args = create_sft_config_from_yaml(config, data_args)
    
    # Create PEFT configuration if enabled
    peft_config = None
    if config.get("peft", {}).get("enabled", False):
        peft_config = create_peft_config(config.get("peft", {}))
        logging.info(f"Using PEFT method: {config['peft']['method']}")
    
    # Get model name/path
    model_name_or_path = data_args.model_name_or_path or config.get("training", {}).get("model_name_or_path", "Qwen/Qwen2-Audio-7B-Instruct")
    
    # Get TensorBoard settings from config or command line args
    enable_tensorboard = config.get("training", {}).get("enable_tensorboard", True)
    tensorboard_log_dir = config.get("training", {}).get("tensorboard_log_dir", None)
    
    # Override with command line arguments if provided
    if data_args.enable_tensorboard is not None:
        enable_tensorboard = data_args.enable_tensorboard
    if data_args.tensorboard_log_dir is not None:
        tensorboard_log_dir = data_args.tensorboard_log_dir
    
    trainer = SFTTrainer(
        model=model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=peft_config,
        enable_tensorboard=enable_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main() 