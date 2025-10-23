# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Optional, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from trl.data_utils import maybe_apply_chat_template
from trl.trainer.sft_config import SFTConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) of language models. This trainer is specifically designed
    for audio question answering tasks and supports both text and audio inputs.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("your_audio_dataset", split="train")

    trainer = SFTTrainer(
        model="Qwen/Qwen2-Audio-7B-Instruct",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include columns `"prompt"` and `"solution"`. Any additional columns 
            in the dataset are ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: SFTConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        enable_tensorboard: bool = True,
        tensorboard_log_dir: Optional[str] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            if "Qwen2-Audio" in model_id:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()

        # Processing class
        if processing_class is None:
            if "Qwen2-Audio" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Data collator
        def data_collator(features):
            # Extract prompts and solutions
            prompts = [feature["prompt"] for feature in features]
            solutions = [feature["solution"] for feature in features]
            audios = [feature.get("audio", None) for feature in features]
            
            # Apply chat template if needed
            prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in features]
            
            # Create full text (prompt + solution) for processing
            full_texts = [prompt + solution for prompt, solution in zip(prompts_text, solutions)]
            
            # Process inputs
            if any(audio is not None for audio in audios):
                # Audio + text input
                inputs = processing_class(
                    text=full_texts,
                    audios=audios,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Text-only input
                inputs = processing_class(
                    text=full_texts,
                    return_tensors="pt",
                    padding=True
                )
            
            # Get the processed tokens
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Create labels tensor with the same shape as input_ids
            labels = torch.full_like(input_ids, -100)
            
            # Find where the solution starts by processing prompts separately
            prompt_inputs = processing_class(
                text=prompts_text,
                audios=audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
            
            # Fill in the labels for the solution part
            for i, prompt_length in enumerate(prompt_lengths):
                # Set labels for the solution part (after the prompt)
                labels[i, prompt_length:] = input_ids[i, prompt_length:]
            
            # Set padding tokens to -100
            labels[labels == processing_class.pad_token_id] = -100
            
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            
            # Add audio features if present
            if "input_features" in inputs:
                result["input_features"] = inputs["input_features"]
                result["feature_attention_mask"] = inputs["feature_attention_mask"]
            
            return result

        # Training arguments

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in SFT, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        # Initialize TensorBoard writer
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        if self.enable_tensorboard:
            if tensorboard_log_dir is None:
                tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
            self.tensorboard_writer = SummaryWriter(tensorboard_log_dir)
            print(f"TensorBoard logging enabled. Logs will be saved to: {tensorboard_log_dir}")
        else:
            self.tensorboard_writer = None
            if enable_tensorboard and not TENSORBOARD_AVAILABLE:
                print("Warning: TensorBoard requested but not available. Install with: pip install tensorboard")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize score token IDs for per-label accuracy metrics
        self._initialize_score_token_ids()

        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

    def _initialize_score_token_ids(self):
        """
        Initialize score token IDs for per-label accuracy metrics.
        This is done once during initialization to avoid recomputing in every training step.
        """
        self.score_token_ids = {}
        
        # Get tokenizer to encode score tokens
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            tokenizer = self.processing_class.tokenizer if hasattr(self.processing_class, 'tokenizer') else self.processing_class
        else:
            # Fallback: try to get tokenizer from model
            tokenizer = getattr(self.model, 'tokenizer', None)
            if tokenizer is None:
                print("Warning: Could not find tokenizer for score token initialization")
                return
        
        # Get token IDs for score tokens "1", "2", "3", "4", "5"
        for score in ["1", "2", "3", "4", "5"]:
            try:
                token_ids = tokenizer.encode(score, add_special_tokens=False)
                if token_ids:
                    self.score_token_ids[score] = token_ids[0]  # Take the first token ID
                    print(f"Initialized score token '{score}' -> ID {token_ids[0]}")
            except Exception as e:
                print(f"Warning: Could not encode score token '{score}': {e}")
                continue
        
        if self.score_token_ids:
            print(f"Initialized {len(self.score_token_ids)} score token IDs: {self.score_token_ids}")
        else:
            print("Warning: No score token IDs were initialized")

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In SFTTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "solution"]

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Override to disable length grouping since our dataset doesn't have input_ids.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Disable length grouping for our custom dataset
        if hasattr(self.args, 'group_by_length') and self.args.group_by_length:
            # Temporarily disable group_by_length
            original_group_by_length = self.args.group_by_length
            self.args.group_by_length = False
            dataloader = super().get_train_dataloader()
            # Restore the original setting
            self.args.group_by_length = original_group_by_length
            return dataloader
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        """
        Override to disable length grouping for evaluation as well.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Disable length grouping for our custom dataset
        if hasattr(self.args, 'group_by_length') and self.args.group_by_length:
            # Temporarily disable group_by_length
            original_group_by_length = self.args.group_by_length
            self.args.group_by_length = False
            dataloader = super().get_eval_dataloader(eval_dataset)
            # Restore the original setting
            self.args.group_by_length = original_group_by_length
            return dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SFTTrainer does not support returning outputs")
        
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        # Handle audio features if present
        if "input_features" in inputs:
            input_features = inputs["input_features"]
            feature_attention_mask = inputs["feature_attention_mask"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        loss = outputs.loss
        
        # Log metrics
        if labels is not None:
            # Calculate accuracy (excluding -100 labels)
            # Shift logits and labels to align predictions with targets (causal LM: predict t+1 given t)
            # Exclude -100 labels after shifting
            # logits: (batch, seq_len, vocab), labels: (batch, seq_len)
            # outputs.logits predicts token t+1 at position t
            shifted_logits = outputs.logits[:, :-1, :]  # remove last token prediction (EOS)
            shifted_labels = labels[:, 1:]              # remove first label (BOS)
            shifted_valid_labels = shifted_labels != -100

            if shifted_valid_labels.any():
                # Get predictions for valid positions
                pred_labels = shifted_logits.argmax(dim=-1)[shifted_valid_labels]
                true_labels = shifted_labels[shifted_valid_labels]
                
                # Overall accuracy
                accuracy = (pred_labels == true_labels).float().mean()
                self._metrics["accuracy"].append(accuracy.item())
                
                # Per-label accuracy for score tokens (1, 2, 3, 4, 5)
                self._compute_per_label_accuracy(pred_labels, true_labels)
        
        return loss
    
    def _compute_per_label_accuracy(self, pred_labels, true_labels):
        """
        Compute accuracy for specific target labels (1, 2, 3, 4, 5).
        
        Args:
            pred_labels: Predicted token IDs
            true_labels: True token IDs
        """
        # Use pre-computed score token IDs from initialization
        if not hasattr(self, 'score_token_ids') or not self.score_token_ids:
            return
        
        # Compute accuracy for each score token
        for score, token_id in self.score_token_ids.items():
            # Find positions where true label is this score token
            mask = (true_labels == token_id)
            
            if mask.any():
                # Calculate accuracy for this specific score
                correct_predictions = (pred_labels[mask] == true_labels[mask]).float().sum()
                total_predictions = mask.float().sum()
                label_accuracy = correct_predictions / total_predictions
                
                # Store the metric
                metric_name = f"accuracy_score_{score}"
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = []
                self._metrics[metric_name].append(label_accuracy.item())
                
                # Also store count for debugging
                count_name = f"count_score_{score}"
                if count_name not in self._metrics:
                    self._metrics[count_name] = []
                self._metrics[count_name].append(total_predictions.item())

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Separate accuracy metrics from count metrics
        accuracy_metrics = {}
        count_metrics = {}
        other_metrics = {}
        
        for key, val in self._metrics.items():
            if key.startswith('accuracy_score_'):
                # Average accuracy metrics
                accuracy_metrics[key] = sum(val) / len(val) if val else 0.0
            elif key.startswith('count_score_'):
                # Sum count metrics (total occurrences)
                count_metrics[key] = sum(val) if val else 0.0
            else:
                # Average other metrics
                other_metrics[key] = sum(val) / len(val) if val else 0.0
        
        # Combine all metrics
        metrics = {**accuracy_metrics, **count_metrics, **other_metrics}
        logs = {**logs, **metrics}
        
        # Log to TensorBoard if enabled
        if self.enable_tensorboard and self.tensorboard_writer is not None:
            global_step = self.state.global_step if hasattr(self.state, 'global_step') else 0
            
            # Log all metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, global_step)
            
            # Log learning rate
            if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
                lr = self.optimizer.param_groups[0]['lr']
                self.tensorboard_writer.add_scalar('learning_rate', lr, global_step)
            
            # Log aggregated score accuracy (average of all score accuracies)
            score_accuracies = [v for k, v in accuracy_metrics.items() if k.startswith('accuracy_score_')]
            if score_accuracies:
                avg_score_accuracy = sum(score_accuracies) / len(score_accuracies)
                self.tensorboard_writer.add_scalar('avg_score_accuracy', avg_score_accuracy, global_step)
            
            # Flush the writer to ensure logs are written
            self.tensorboard_writer.flush()
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    
    def __del__(self):
        """Clean up TensorBoard writer when trainer is destroyed."""
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None:
            self.tensorboard_writer.close() 