# Copyright 2025 Jiatong Shi (Anuttacon)
from .base_audiollm import BaseAudioLLM
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List, Optional
import torch
import json
import librosa
import re
import time
import torch.nn.functional as F

class Qwen2Audio(BaseAudioLLM):
    def __init__(self, model_name: str, model_tag: str, system_prompt: str, temperature: float = 1.5):
        self.model_name = model_name
        self.model_tag = model_tag
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_tag,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.system_prompt = system_prompt
        self.temperature = temperature
        
        self.processor = AutoProcessor.from_pretrained(model_tag)
        
        # Retry settings
        self.max_retries = 10
        self.initial_delay = 0
        self.backoff_factor = 1.5
        
        # Scoring token IDs (1-5 scores) - may need adjustment for Qwen2-Audio
        self.score_token_ids = [16, 17, 18, 19, 20]  # Token IDs for scores 1-5
        
        # Content pass token IDs (0-1 scores) for content_pass prediction
        self.content_pass_token_ids = [15, 16]  # Token IDs for scores 0-1

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

    def _get_score_from_token_probabilities(self, inputs, is_content_pass: bool = False) -> float:
        """
        Get the score by analyzing token probabilities for scoring tokens.
        
        Args:
            inputs: Model inputs
            is_content_pass: If True, use 0-1 tokens; if False, use 1-5 tokens
            
        Returns:
            float: Best score based on token probabilities (0-1 for content_pass, 1-5 for regular scoring)
        """
        with torch.no_grad():
            # Use generate method with return_dict_in_generate to get logits
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Only generate 1 token to get logits
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,  # Deterministic for logit analysis
            )
            
            # Get the logits from the generated scores
            if hasattr(outputs, 'scores') and outputs.scores:
                # Get logits from the first (and only) generated token
                logits = outputs.scores[0][0]  # Shape: [vocab_size]
                
                if is_content_pass:
                    # Use 0-1 tokens for content_pass prediction
                    score_token_tensor = torch.tensor(self.content_pass_token_ids, device=logits.device, dtype=torch.long)
                    score_logits = logits[score_token_tensor]  # Shape: [2]
                    
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
                    score_token_tensor = torch.tensor(self.score_token_ids, device=logits.device, dtype=torch.long)
                    score_logits = logits[score_token_tensor]  # Shape: [5]

                    # Apply softmax to get probabilities
                    score_probs = F.softmax(score_logits, dim=-1)
                    
                    expected_score = torch.dot(score_probs, torch.arange(1, 6, device=logits.device).float())
                    print(f"Expected score: {expected_score}")
                    
                    return float(expected_score)
            else:
                # Fallback: generate normally and parse response
                print("Warning: Could not get logits, falling back to text generation")
                return self._fallback_generate_score(inputs, is_content_pass)
    
    def _fallback_generate_score(self, inputs, is_content_pass: bool = False) -> float:
        """Fallback method to generate score from text response"""
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=self.temperature,
            top_k=30,
        )
        
        text = self.processor.decode(output[0], skip_special_tokens=True)
        response = self._parse_assistant_response(text)
        
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
                # Load and prepare audio
                waveform, sample_rate = librosa.load(audio_path, sr=16000)
                
                if only_message:
                    # For Qwen2-Audio, we return the formatted prompt with audio markers
                    formatted_prompt = f"{self.system_prompt}\n\n{prompt}\n\n<|audio_bos|><|AUDIO|><|audio_eos|>"
                    return formatted_prompt
                
                # Prepare model input - ensure audio tokens are in the text
                # According to Qwen2Audio docs, the text should contain <|AUDIO|> token
                prompt_with_audio = f"{prompt}\n\n<|audio_bos|><|AUDIO|><|audio_eos|>"
                
                inputs = self.processor(
                    text=prompt_with_audio, 
                    audios=waveform, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).to(self.model.device)

                # Use token probability analysis instead of generation
                response = self._get_score_from_token_probabilities(inputs, is_content_pass)
                
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
                        
                        time.sleep(delay)
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
                        
                        time.sleep(delay)
                        delay *= self.backoff_factor
                    
            except Exception as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    print(f"Failed after {self.max_retries} retries. Last error: {str(e)}")
                    raise e
                
                print(f"Error during generation: {str(e)}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                time.sleep(delay)
                delay *= self.backoff_factor

    def generate(self, audio_path: str, prompt: str, only_message: bool = False) -> Any:
        """
        Process the audio and prompt for inference with float validation retry.
        """
        return self._generate_with_retry(audio_path, prompt, only_message)

    def batch_generate(self, prompts: List[str], audio_paths: List[str], num_workers: int = 4, is_content_pass: bool = False) -> List[Dict[str, Any]]:
        """Generate responses for multiple audio files with token probability analysis"""
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
            # For Qwen2-Audio, the response might not have the assistant marker
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