"""
Utility functions for PEFT configuration.
"""
from typing import Dict, Any, Optional
from peft import (
    LoraConfig,
    AdaLoraConfig,
    IA3Config,
    PrefixTuningConfig,
    PromptTuningConfig,
    PeftConfig,
)


def create_peft_config(config: Dict[str, Any]) -> Optional[PeftConfig]:
    """
    Create a PEFT configuration based on the provided config dictionary.
    
    Args:
        config: Dictionary containing PEFT configuration parameters
        
    Returns:
        PeftConfig object or None if PEFT is disabled
    """
    if not config.get("enabled", False):
        return None
    
    method = config.get("method", "lora").lower()
    
    if method == "lora":
        return _create_lora_config(config.get("lora", {}))
    elif method == "adalora":
        return _create_adalora_config(config.get("adalora", {}))
    elif method == "ia3":
        return _create_ia3_config(config.get("ia3", {}))
    elif method == "prefix_tuning":
        return _create_prefix_tuning_config(config.get("prefix_tuning", {}))
    elif method == "prompt_tuning":
        return _create_prompt_tuning_config(config.get("prompt_tuning", {}))
    else:
        raise ValueError(f"Unsupported PEFT method: {method}")


def _create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=config.get("r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=config.get("lora_dropout", 0.1),
        bias=config.get("bias", "none"),
        task_type=config.get("task_type", "CAUSAL_LM"),
    )


def _create_adalora_config(config: Dict[str, Any]) -> AdaLoraConfig:
    """Create AdaLoRA configuration."""
    return AdaLoraConfig(
        r=config.get("r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=config.get("lora_dropout", 0.1),
        bias=config.get("bias", "none"),
        task_type=config.get("task_type", "CAUSAL_LM"),
        init_r=config.get("init_r", 12),
        target_r=config.get("target_r", 8),
        beta1=config.get("beta1", 0.85),
        beta2=config.get("beta2", 0.85),
        tinit=config.get("tinit", 200),
        tfinal=config.get("tfinal", 1000),
        deltaT=config.get("deltaT", 10),
        orth_reg_weight=config.get("orth_reg_weight", 0.42),
    )


def _create_ia3_config(config: Dict[str, Any]) -> IA3Config:
    """Create IA3 configuration."""
    return IA3Config(
        target_modules=config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        feedforward_modules=config.get("feedforward_modules", ["gate_proj", "up_proj", "down_proj"]),
        task_type=config.get("task_type", "CAUSAL_LM"),
    )


def _create_prefix_tuning_config(config: Dict[str, Any]) -> PrefixTuningConfig:
    """Create Prefix Tuning configuration."""
    return PrefixTuningConfig(
        num_virtual_tokens=config.get("num_virtual_tokens", 20),
        encoder_hidden_size=config.get("encoder_hidden_size", 128),
        prefix_projection=config.get("prefix_projection", False),
        task_type=config.get("task_type", "CAUSAL_LM"),
    )


def _create_prompt_tuning_config(config: Dict[str, Any]) -> PromptTuningConfig:
    """Create Prompt Tuning configuration."""
    return PromptTuningConfig(
        num_virtual_tokens=config.get("num_virtual_tokens", 20),
        encoder_hidden_size=config.get("encoder_hidden_size", 128),
        task_type=config.get("task_type", "CAUSAL_LM"),
    ) 