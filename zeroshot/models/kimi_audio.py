# Copyright 2025 Jiatong Shi (Anuttacon)
from .base_audiollm import BaseAudioLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Union
import torch
import json
import librosa
import re
import time
import torch.nn.functional as F
import numpy as np
import os
import tqdm
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from huggingface_hub import snapshot_download

# Import Whisper components (these are available)
from .kimiaudio_utility.whisper_Lv3.whisper import WhisperEncoder
from .kimiaudio_utility.glm4_tokenizer import Glm4Tokenizer

class KimiAContent:
    """KimiAContent class following the official Kimi-Audio implementation"""
    def __init__(self):
        self.audio_tokens = []
        self.text_tokens = []
        self.continuous_feature = []
        self.audio_token_loss_mask = []
        self.text_token_loss_mask = []
    
    def audio_append(self, token, audio_token_loss_mask=True):
        if isinstance(token, int):
            self.audio_tokens.append(token)
            self.audio_token_loss_mask.append(audio_token_loss_mask)
        else:
            raise ValueError(f"Token must be int, got {type(token)}")
    
    def audio_extend(self, tokens, is_continuous=False, audio_token_loss_mask=True):
        if isinstance(tokens, list):
            self.audio_tokens.extend(tokens)
            if is_continuous:
                # For continuous audio tokens, only the last token has loss mask
                self.audio_token_loss_mask.extend([False] * (len(tokens) - 1) + [audio_token_loss_mask])
            else:
                self.audio_token_loss_mask.extend([audio_token_loss_mask] * len(tokens))
        else:
            raise ValueError(f"Tokens must be list, got {type(tokens)}")
    
    def text_append(self, token, has_loss=True):
        if isinstance(token, int):
            self.text_tokens.append(token)
            self.text_token_loss_mask.append(has_loss)
        else:
            raise ValueError(f"Token must be int, got {type(token)}")
    
    def text_extend(self, tokens, has_loss=True):
        if isinstance(tokens, list):
            self.text_tokens.extend(tokens)
            self.text_token_loss_mask.extend([has_loss] * len(tokens))
        else:
            raise ValueError(f"Tokens must be list, got {type(tokens)}")
    
    def merge(self, other):
        """Merge another KimiAContent into this one"""
        self.audio_tokens.extend(other.audio_tokens)
        self.text_tokens.extend(other.text_tokens)
        self.audio_token_loss_mask.extend(other.audio_token_loss_mask)
        self.text_token_loss_mask.extend(other.text_token_loss_mask)
        if other.continuous_feature:
            self.continuous_feature.extend(other.continuous_feature)
    
    def is_valid(self):
        """Check if the content is valid"""
        return (len(self.audio_tokens) == len(self.audio_token_loss_mask) and 
                len(self.text_tokens) == len(self.text_token_loss_mask))
    
    def to_tensor(self):
        """Convert to tensors following the official implementation"""
        audio_tokens = torch.tensor(self.audio_tokens, dtype=torch.long).unsqueeze(0)
        text_tokens = torch.tensor(self.text_tokens, dtype=torch.long).unsqueeze(0)
        is_continuous_mask = torch.tensor(self.audio_token_loss_mask, dtype=torch.bool).unsqueeze(0)
        text_loss_mask = torch.tensor(self.text_token_loss_mask, dtype=torch.bool).unsqueeze(0)
        continuous_feature = self.continuous_feature if self.continuous_feature else []
        
        return audio_tokens, text_tokens, is_continuous_mask, text_loss_mask, continuous_feature

class KimiAPromptManager:
    """KimiAPromptManager following the official implementation"""
    def __init__(self, model_path: str, kimia_token_offset: int, kimia_text_audiodelaytokens: int):
        # Initialize audio tokenizer first
        self.audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        self.audio_tokenizer = self.audio_tokenizer.to(torch.cuda.current_device())

        logger.info(f"Looking for resources in {model_path}")
        logger.info(f"Loading whisper model")

        # Initialize whisper encoder
        self.whisper_model = WhisperEncoder(
            os.path.join(model_path, "whisper-large-v3"), mel_batch_size=20
        )
        self.whisper_model = self.whisper_model.to(torch.cuda.current_device())
        self.whisper_model = self.whisper_model.bfloat16()
        self.whisper_model.eval()

        # Load text tokenizer
        logger.info(f"Loading text tokenizer")
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            logger.info(f"Can not find text tokenizer in {model_path}, Loading default text tokenizer from moonshotai/Kimi-Audio-7B-Instruct")
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-Audio-7B-Instruct", trust_remote_code=True
            )

        # Get extra tokens
        self.extra_tokens = self._instantiate_extra_tokens()

        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset
    
    def _instantiate_extra_tokens(self):
        """Instantiate extra tokens following the official implementation"""
        class ExtraTokens:
            def __init__(self, tokenizer):
                # Try to find special tokens in the tokenizer
                self.kimia_user_msg_start = tokenizer.encode("<|user|>", bos=False, eos=False)[0] if tokenizer.encode("<|user|>", bos=False, eos=False) else 1
                self.kimia_assistant_msg_start = tokenizer.encode("<|assistant|>", bos=False, eos=False)[0] if tokenizer.encode("<|assistant|>", bos=False, eos=False) else 2
                self.kimia_text_blank = tokenizer.encode("<|text_blank|>", bos=False, eos=False)[0] if tokenizer.encode("<|text_blank|>", bos=False, eos=False) else 0
                self.kimia_text_eos = tokenizer.encode("<|text_eos|>", bos=False, eos=False)[0] if tokenizer.encode("<|text_eos|>", bos=False, eos=False) else 3
                self.media_begin = tokenizer.encode("<|media_begin|>", bos=False, eos=False)[0] if tokenizer.encode("<|media_begin|>", bos=False, eos=False) else 4
                self.media_end = tokenizer.encode("<|media_end|>", bos=False, eos=False)[0] if tokenizer.encode("<|media_end|>", bos=False, eos=False) else 5
                self.kimia_speech_ct_id = tokenizer.encode("<|speech_ct|>", bos=False, eos=False)[0] if tokenizer.encode("<|speech_ct|>", bos=False, eos=False) else 6
                self.kimia_speech_ctd_id = tokenizer.encode("<|speech_ctd|>", bos=False, eos=False)[0] if tokenizer.encode("<|speech_ctd|>", bos=False, eos=False) else 7
                self.msg_end = tokenizer.encode("<|msg_end|>", bos=False, eos=False)[0] if tokenizer.encode("<|msg_end|>", bos=False, eos=False) else 8
        
        return ExtraTokens(self.text_tokenizer)
    
    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids
    
    def _tokenize_audio(self, wav_path):
        """Audio tokenization using the audio tokenizer"""
        wav_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
        wav_tokens = wav_tokens + self.kimia_token_offset
        wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
        return wav_tokens_list

    def extract_whisper_feat(self, wav: Union[torch.Tensor, str]):
        """Whisper feature extraction using WhisperEncoder"""
        if isinstance(wav, str):
            wav_array = librosa.load(wav, sr=16000)[0]
            wav_tensor = torch.tensor(wav_array).unsqueeze(0)[:, :]
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav
        else:
            raise ValueError(f"Invalid wav type: {type(wav)}")
        
        assert self.whisper_model is not None
        wav_tensor = wav_tensor.to(torch.cuda.current_device())
        continous_feature = self.whisper_model.tokenize_waveform(wav_tensor)
        continous_feature = continous_feature.reshape(
            continous_feature.shape[0],
            int(continous_feature.shape[1] // 4),
            continous_feature.shape[2] * 4,
        )
        return continous_feature

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        output_type: str = "text",
    ):
        kimia_content_msg = KimiAContent()

        role = message["role"]
        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            if text_tokens is not None:
                kimia_content_msg.text_extend(text_tokens, has_loss)
                kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)  # eos for text stream
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif message["message_type"] == "audio":
            if "audio_tokens" in message:
                speech_tokens = message["audio_tokens"]
            else:
                audio_path = message["content"]
                speech_tokens = self._tokenize_audio(audio_path)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)  # EOS for audio stream
            kimia_content_msg.text_extend([self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2))

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ctd_id)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(audio_path)
                kimia_content_msg.continuous_feature.append(whisper_feature)

        elif message["message_type"] == "audio-text":
            audio_path, text = message["content"]
            speech_tokens = self._tokenize_audio(audio_path)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * self.kimia_text_audiodelaytokens)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            if text_tokens is not None:
                kimia_content_msg.text_extend(text_tokens)
                text_pad_tokens = (self.kimia_text_audiodelaytokens + len(speech_tokens) - len(text_tokens)) * [self.extra_tokens.kimia_text_blank]
                kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] is None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert kimia_content_msg.is_valid(), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def get_prompt(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAContent:
        """
        messages: List[Dict]
        messages[i] = {
            "role": "user" | "assistant" | "system",
            "content": str
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                extract_whisper_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                message={
                    "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )
            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]
        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg

class KimiASampler:
    """Simplified KimiASampler following the official implementation"""
    def __init__(self, audio_top_k=5, audio_temperature=0.0, audio_repetition_penalty=1.0, 
                 audio_repetition_window_size=64, text_top_k=5, text_temperature=0.0, 
                 text_repetition_penalty=1.0, text_repetition_window_size=16):
        self.audio_top_k = audio_top_k
        self.audio_temperature = audio_temperature
        self.audio_repetition_penalty = audio_repetition_penalty
        self.audio_repetition_window_size = audio_repetition_window_size
        self.text_top_k = text_top_k
        self.text_temperature = text_temperature
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size
    
    def sample_text_logits(self, text_logits, recent_tokens=None):
        """Sample text token"""
        if recent_tokens is not None and self.text_repetition_penalty != 1.0:
            # Apply repetition penalty
            for token_id in recent_tokens[-self.text_repetition_window_size:]:
                text_logits[0, token_id] /= self.text_repetition_penalty
        
        if self.text_temperature == 0:
            # Greedy sampling
            next_token = torch.argmax(text_logits[0])
        else:
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(text_logits[0], self.text_top_k)
            probs = F.softmax(top_k_logits / self.text_temperature, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_token_idx]
        
        return next_token
    
    def sample_audio_logits(self, audio_logits, recent_tokens=None):
        """Sample audio token"""
        if recent_tokens is not None and self.audio_repetition_penalty != 1.0:
            # Apply repetition penalty
            for token_id in recent_tokens[-self.audio_repetition_window_size:]:
                audio_logits[0, token_id] /= self.audio_repetition_penalty
        
        if self.audio_temperature == 0:
            # Greedy sampling
            next_token = torch.argmax(audio_logits[0])
        else:
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(audio_logits[0], self.audio_top_k)
            probs = F.softmax(top_k_logits / self.audio_temperature, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_token_idx]
        
        return next_token

class KimiAudio(BaseAudioLLM):
    def __init__(self, model_name: str, model_tag: str, system_prompt: str, temperature: float = 1.5):
        self.model_name = model_name
        self.model_tag = model_tag
        self.system_prompt = system_prompt
        self.temperature = temperature
        
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_tag):
            # local path
            cache_path = model_tag
        else:
            # cache everything if model_tag is a model-id
            cache_path = snapshot_download(model_tag)
    
        logger.info(f"Looking for resources in {cache_path}")
        
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_text_audiodelaytokens = getattr(model_config, 'kimia_mimo_audiodelaytokens', 0)
        self.kimia_token_offset = getattr(model_config, 'kimia_token_offset', 32000)

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, 
            kimia_token_offset=self.kimia_token_offset, 
            kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens
        )

        # Whisper encoder and audio tokenizer are already initialized in the prompt manager

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]
        
        # Retry settings
        self.max_retries = 10
        self.initial_delay = 0
        self.backoff_factor = 1.5
        
        # Find token IDs for numbers 1-5 (for regular scoring)
        self.score_token_ids = []
        for i in range(1, 6):
            token_id = self.prompt_manager.text_tokenizer.encode(str(i), bos=False, eos=False)
            if token_id:
                self.score_token_ids.append(token_id[0])
        
        # Find token IDs for numbers 0 and 1 (for content_pass prediction)
        self.content_pass_token_ids = []
        for i in range(0, 2):
            token_id = self.prompt_manager.text_tokenizer.encode(str(i), bos=False, eos=False)
            if token_id:
                self.content_pass_token_ids.append(token_id[0])

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: Optional[torch.Tensor] = None,
        continous_feature: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_type: str = "text",
    ):
        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (max_new_tokens,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (max_new_tokens,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone() if text_input_ids is not None else None
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        for i in tqdm.tqdm(
            range(max_new_tokens), desc="Generating tokens", disable=False
        ):
            try:
                audio_logits, text_logits, past_key_values = self.alm.forward(
                    input_ids=decoder_input_audio_ids,
                    text_input_ids=decoder_input_text_ids,
                    whisper_input_feature=decoder_input_whisper_feature,
                    is_continuous_mask=decoder_is_continuous_mask,
                    position_ids=decoder_position_ids,
                    past_key_values=past_key_values,
                    return_dict=False,
                )

                # Sample text token using the sampler
                recent_text_tokens = text_previous_tokens[:i] if i > 0 else None
                next_token_text = sampler.sample_text_logits(
                    text_logits, recent_tokens=recent_text_tokens
                )

                # Sample audio token using the sampler
                recent_audio_tokens = previous_audio_tokens[:i] if i > 0 else None
                next_audio_token = sampler.sample_audio_logits(
                    audio_logits, recent_tokens=recent_audio_tokens
                )

                if text_stream_is_finished:
                    next_token_text.fill_(self.extra_tokens.kimia_text_blank)
                elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                    text_stream_is_finished = True
                else:
                    valid_text_length += 1

                text_previous_tokens[i : i + 1] = next_token_text

                if i < self.kimia_text_audiodelaytokens:
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    if output_type == "text":
                        next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                    else:
                        valid_audio_length += 1

                previous_audio_tokens[i : i + 1] = next_audio_token

                audio_stream_is_finished = next_audio_token.item() in self.eod_ids

                if (
                    output_type == "text"
                    and text_stream_is_finished
                    or output_type == "both"
                    and audio_stream_is_finished
                ):
                    return_text_tokens = (
                        text_previous_tokens[:valid_text_length]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    return_audio_tokens = (
                        previous_audio_tokens[
                            self.kimia_text_audiodelaytokens : valid_audio_length
                            + self.kimia_text_audiodelaytokens
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    return return_audio_tokens, return_text_tokens
                else:
                    decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                    decoder_input_text_ids = next_token_text.unsqueeze(1)

                    decoder_position_ids = (
                        torch.zeros(1, 1, device=torch.cuda.current_device())
                        .fill_(last_position_id + 1)
                        .long()
                        .view(1, 1)
                    )
                    last_position_id += 1

                    decoder_input_whisper_feature = None
                    decoder_is_continuous_mask = None
                    
            except Exception as e:
                logger.error(f"Error in generation loop at step {i}: {e}")
                import traceback
                traceback.print_exc()
                exit(0)
                # Return what we have so far
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)

        return None, generated_text  # Return None for audio since we don't have detokenizer

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

    def _validate_float_response(self, response: Any) -> bool:
        """Validate if response is a valid float number"""
        try:
            if response is None:
                return False
            elif isinstance(response, (int, float)):
                return True
            elif isinstance(response, str):
                # Try to convert string to float
                float(response.strip())
                return True
            else:
                return False
        except (ValueError, TypeError):
            return False

    def _get_score_from_token_probabilities(self, audio_input_ids: torch.Tensor, text_input_ids: torch.Tensor, 
                                          is_continuous_mask: torch.Tensor, continous_feature: List[torch.Tensor],
                                          is_content_pass: bool = False) -> float:
        """
        Get the score by analyzing token probabilities for scoring tokens.
        
        Args:
            audio_input_ids: Audio input tensor
            text_input_ids: Text input tensor
            is_continuous_mask: Continuous mask tensor
            continous_feature: Whisper features
            is_content_pass: If True, use 0-1 tokens; if False, use 1-5 tokens
            
        Returns:
            float: Best score based on token probabilities (0-1 for content_pass, 1-5 for regular scoring)
        """
        try:
            with torch.no_grad():
                # Forward pass to get logits
                audio_logits, text_logits, _ = self.alm.forward(
                    input_ids=audio_input_ids,
                    text_input_ids=text_input_ids,
                    whisper_input_feature=continous_feature,
                    is_continuous_mask=is_continuous_mask,
                    return_dict=False,
                )
                
                # Get text logits for the last position
                last_text_logits = text_logits[0, -1]  # Shape: [vocab_size]
                
                if is_content_pass:
                    # Use 0-1 tokens for content_pass prediction
                    score_token_tensor = torch.tensor(self.content_pass_token_ids, device=last_text_logits.device, dtype=torch.long)
                    score_logits = last_text_logits[score_token_tensor]  # Shape: [2]
                    
                    # Apply softmax to get probabilities
                    score_probs = F.softmax(score_logits, dim=-1)
                    
                    # For content_pass, return the probability of class 1 (pass)
                    # If probability of 1 is higher than 0.5, return 1.0, otherwise return 0.0
                    prob_pass = score_probs[1].item()  # Probability of class 1 (pass)
                    result = 1.0 if prob_pass >= 0.5 else 0.0
                    print(f"Content pass probabilities: [fail={score_probs[0]:.3f}, pass={score_probs[1]:.3f}], result={result}")
                    
                    return result
                else:
                    # Use 1-5 tokens for regular scoring
                    score_token_tensor = torch.tensor(self.score_token_ids, device=last_text_logits.device, dtype=torch.long)
                    score_logits = last_text_logits[score_token_tensor]  # Shape: [5]

                    # Apply softmax to get probabilities
                    score_probs = F.softmax(score_logits, dim=-1)
                    
                    # Calculate expected score
                    expected_score = torch.dot(score_probs.float(), torch.arange(1, 6, device=last_text_logits.device).float())
                    print(f"Expected score: {expected_score}")
                    
                    return float(expected_score)
        except Exception as e:
            print(f"Warning: Could not get token probabilities: {e}, falling back to text generation")
            return self._fallback_generate_score(audio_input_ids, text_input_ids, is_continuous_mask, continous_feature, is_content_pass)
    
    def _fallback_generate_score(self, audio_input_ids: torch.Tensor, text_input_ids: torch.Tensor,
                                is_continuous_mask: torch.Tensor, continous_feature: List[torch.Tensor],
                                is_content_pass: bool = False) -> float:
        """Fallback method to generate score from text response"""
        # Generate text response
        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=256,
            is_continuous_mask=is_continuous_mask,
            continous_feature=continous_feature,
            output_type="text",
        )
        
        generated_text = self.detokenize_text(generated_text_tokens)
        response = self._parse_assistant_response(generated_text)
        
        # Try to convert to float and validate range
        if is_content_pass:
            # For content_pass, expect 0 or 1
            if isinstance(response, (int, float)) and response in [0, 1]:
                return float(response)
            else:
                raise ValueError(f"Invalid content_pass response: {response}")
        else:
            # For regular scoring, expect 1-5
            if isinstance(response, (int, float)) and 1 <= response <= 5:
                return float(response)
            else:
                raise ValueError(f"Invalid response: {response}")

    def _generate_with_retry(self, audio_path: str, prompt: str, only_message: bool = False, is_content_pass: bool = False) -> Any:
        """Generate response with retry logic for float validation"""
        
        retry_count = 0
        delay = self.initial_delay
        
        while True:
            try:
                if only_message:
                    # Return the formatted prompt with audio markers
                    formatted_prompt = f"{self.system_prompt}\n\n{prompt}\n\n<|audio_bos|><|AUDIO|><|audio_eos|>"
                    return formatted_prompt
                
                # Create messages in Kimi-Audio format
                chats = [
                    # You can provide context or instructions as text
                    {"role": "user", "message_type": "text", "content": prompt},
                    # Provide the audio file path
                    {"role": "user", "message_type": "audio", "content": audio_path}
                ]
                
                # Get prompt tensors
                history = self.prompt_manager.get_prompt(chats, output_type="text")
                audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
                audio_features = history.continuous_feature

                # Move to device
                audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
                text_input_ids = text_input_ids.to(torch.cuda.current_device())
                is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
                audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

                # Use token probability analysis instead of generation
                response = self._get_score_from_token_probabilities(
                    audio_input_ids, text_input_ids, is_continuous_mask, audio_features, is_content_pass
                )
                
                # Validate the response based on mode
                if is_content_pass:
                    # For content_pass, expect 0 or 1
                    if response in [0.0, 1.0]:
                        return response
                    else:
                        # Invalid response, retry
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            print(f"Failed to get valid content_pass response after {self.max_retries} retries. Returning last response.")
                            return response if response in [0.0, 1.0] else 1.0  # Default to pass
                        
                        print(f"Invalid content_pass response received: {response}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                        
                        delay *= self.backoff_factor
                else:
                    # For regular scoring, expect 1-5
                    if 1 <= response <= 5:
                        return response
                    else:
                        # Invalid score, retry
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            print(f"Failed to get valid score after {self.max_retries} retries. Returning last response.")
                            return response if 1 <= response <= 5 else 3.0  # Default to middle score
                        
                        print(f"Invalid score received: {response}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                        
                        delay *= self.backoff_factor
                    
            except Exception as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    print(f"Failed after {self.max_retries} retries. Last error: {str(e)}")
                    raise e
                
                print(f"Error during generation: {str(e)}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                delay *= self.backoff_factor

    def batch_generate(self, prompts: List[str], audio_paths: List[str], num_workers: int = 4, is_content_pass: bool = False) -> List[Dict[str, Any]]:
        """Generate responses for multiple audio files using token probability analysis"""
        if len(prompts) != len(audio_paths):
            raise ValueError("Number of prompts must match number of audio paths")
        
        results = []
        for i, (prompt, audio_path) in enumerate(zip(prompts, audio_paths)):
            print(f"Processing {i+1}/{len(prompts)}: {audio_path}")
            
            try:
                response = self._generate_with_retry(audio_path, prompt, is_content_pass=is_content_pass)
                
                # Pack result
                result = {
                    "response": str(response),
                    "audio_path": audio_path,
                    "prompt": prompt,
                    "model_name": self.model_name,
                    "error": False
                }
                
            except Exception as e:
                result = {
                    "response": f"Error: {str(e)}",
                    "audio_path": audio_path,
                    "prompt": prompt,
                    "model_name": self.model_name,
                    "error": True
                }
            
            results.append(result)
        
        return results

    def evaluate_with_schema(self, audio_path: str, prompt: str, is_content_pass: bool = False) -> Dict[str, Any]:
        """Evaluate audio using token probability analysis - similar to other models"""
        try:
            response = self._generate_with_retry(audio_path, prompt, is_content_pass=is_content_pass)
            return {
                "response": str(response),
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name,
                "error": False
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name,
                "error": True
            }
    
    def predict_content_pass(self, audio_path: str, prompt: str) -> float:
        """
        Convenience method to predict content pass (0 or 1) for an audio file.
        
        Args:
            audio_path: Path to the audio file
            prompt: The evaluation prompt
            
        Returns:
            float: 0.0 for fail, 1.0 for pass
        """
        response = self._generate_with_retry(audio_path, prompt, is_content_pass=True)
        return float(response)
    
    def predict_score(self, audio_path: str, prompt: str) -> float:
        """
        Convenience method to predict regular score (1-5) for an audio file.
        
        Args:
            audio_path: Path to the audio file
            prompt: The evaluation prompt
            
        Returns:
            float: Score between 1.0 and 5.0
        """
        response = self._generate_with_retry(audio_path, prompt, is_content_pass=False)
        return float(response)
    
    def _parse_assistant_response(self, text: str) -> Optional[Any]:
        """
        Robustly parse the content after 'assistant\\n' in a text string to extract a float number.
        
        Args:
            text (str): The input text containing the assistant response
            
        Returns:
            float: Parsed float number if successful, None if parsing fails
            
        Raises:
            ValueError: If no assistant response is found in the text
        """
        # Handle list input (convert to string if it's a single-item list)
        if isinstance(text, list) and len(text) == 1:
            text = text[0]
        elif isinstance(text, list):
            raise ValueError("Input list contains multiple items, expected single string")
        
        # Find the assistant response marker
        assistant_marker = "assistant\n"
        assistant_pos = text.find(assistant_marker)
        
        if assistant_pos == -1:
            # For Kimi-Audio, the response might not have the assistant marker
            # Try to extract the last part of the response
            content = text.strip()
        else:
            # Extract content after the marker
            content = text[assistant_pos + len(assistant_marker):].strip()
        
        if not content:
            raise ValueError("No content found in response")
        
        # Try to extract float number from content
        float_value = self._extract_float_content(content)
        
        try:
            return float(float_value)
        except ValueError as e:
            raise ValueError(f"Failed to parse float: {e}")

    def _extract_float_content(self, content: str) -> str:
        """
        Extract float number from content, handling various formatting scenarios.
        
        Args:
            content (str): Raw content that may contain a float number
            
        Returns:
            str: Cleaned float string
        """
        content = content.strip()
        
        # Pattern 1: Direct float number (e.g., "3.5", "4", "2.0")
        float_pattern = r'\b\d+\.?\d*\b'
        float_match = re.search(float_pattern, content)
        if float_match:
            return float_match.group(0)
        
        # Pattern 2: Float in quotes (e.g., "3.5", '4.0')
        quoted_float_pattern = r'["\'](\d+\.?\d*)["\']'
        quoted_match = re.search(quoted_float_pattern, content)
        if quoted_match:
            return quoted_match.group(1)
        
        # Pattern 3: Float in code blocks
        code_block_pattern = r'```\s*\n?(\d+\.?\d*)\n?```'
        code_match = re.search(code_block_pattern, content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Pattern 4: Try to find any number in the content
        number_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(number_pattern, content)
        if numbers:
            return numbers[0]  # Return the first number found
        
        # If no patterns match, return original content and let float parser handle the error
        return content 