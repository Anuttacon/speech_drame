from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAudioLLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str, model_tag: str):
        pass

    @abstractmethod
    def generate(self, prompt: str, audio_path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], audio_paths: List[str]) -> List[Dict[str, Any]]:
        pass
