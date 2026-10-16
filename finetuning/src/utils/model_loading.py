"""Load a full Qwen2-Audio checkpoint or explicitly attach a local PEFT adapter."""
import json
from pathlib import Path


def load_model_and_processor(checkpoint, base_model=None):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    path = Path(checkpoint)
    adapter = path / "adapter_config.json"
    if adapter.is_file():
        config = json.loads(adapter.read_text())
        base = base_model or config.get("base_model_name_or_path")
        if not base:
            raise ValueError("Adapter has no base model; provide --base_model_path")
        from peft import PeftModel
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, str(path))
        processor_source = str(path) if (path / "preprocessor_config.json").is_file() else base
    else:
        if base_model:
            raise ValueError("A base model override applies only to a PEFT adapter")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
        processor_source = checkpoint
    processor = AutoProcessor.from_pretrained(processor_source)
    return model, processor
