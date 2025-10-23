from .base_audiollm import BaseAudioLLM
from typing import Dict, Any, List, Optional
import os
import json
import time
import base64
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
from prompts.schema import EvaluationOutputSchema

class GPT4oAudio(BaseAudioLLM):
    def __init__(self, model_name: str, model_tag: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.model_tag = model_tag
        self.system_prompt = system_prompt or "You are an expert evaluator of speech delivery and storytelling."
        
        # Initialize OpenAI client
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")

        api_endpoint = os.getenv('AZURE_OPENAI_API_ENDPOINT')
        if not api_endpoint:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT environment variable not set")

        self.client = AzureOpenAI(
            api_version="2025-04-01-preview",
            azure_endpoint=api_endpoint,
            api_key=api_key,
        )
        
        # Retry settings
        self.max_retries = 10
        self.initial_delay = 1
        self.backoff_factor = 1.15

    def _encode_audio_file(self, audio_path: str) -> str:
        """Encode audio file to base64 string for OpenAI API"""
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                return audio_base64
        except Exception as e:
            raise ValueError(f"Error encoding audio file {audio_path}: {str(e)}")

    def _ask_with_retry(self, question: str, audio_path: str, output_schema: Any = None) -> str:
        """Ask a question about the audio with retry logic"""
        
        # Encode audio file
        audio_base64 = self._encode_audio_file(audio_path)
        
        retry_count = 0
        delay = self.initial_delay
        
        while True:
            try:
                # Prepare messages with correct format for GPT-4o
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "input_audio", 
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ]
                
                # Prepare request parameters
                params = {
                    "model": self.model_tag,
                    "messages": messages,
                    "max_tokens": 10000,
                    "temperature": 0.1
                }
                
                # Add response format if schema is provided
                # if output_schema:
                #     params["response_format"] = {"type": "json_object"}
                
                # Make API call
                response = self.client.chat.completions.create(**params)
                response_content = response.choices[0].message.content
                
                # Check if response is valid JSON
                try:
                    # Try to parse as JSON first
                    if "```json" in response_content:
                        json_content = response_content.split("```json", 1)[1].split("```", 1)[0].strip()
                        json.loads(json_content)
                    else:
                        json.loads(response_content)
                    
                    # If we reach here, JSON is valid
                    return response_content
                    
                except (json.JSONDecodeError, IndexError, AttributeError):
                    # JSON parsing failed, retry
                    retry_count += 1
                    
                    # Check if retries are exhausted
                    if retry_count > self.max_retries:
                        print(f"Failed to get valid JSON after {self.max_retries} retries. Last response: {response_content[:200]}...")
                        return response_content  # Return the last response even if not valid JSON
                    
                    # Log the retry attempt
                    print(f"Invalid JSON response received. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                    print(f"Response preview: {response_content}")
                    
                    # Wait before retrying with exponential backoff
                    time.sleep(delay)
                    
                    # Increase delay for next retry (exponential backoff)
                    delay *= self.backoff_factor
                    continue
                
            except Exception as e:
                retry_count += 1
                
                # Check if retries are exhausted
                if retry_count > self.max_retries:
                    print(f"Failed after {self.max_retries} retries. Last error: {str(e)}")
                    raise e
                    
                # Log the retry attempt
                print(f"Error during OpenAI API call: {str(e)}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                
                # Wait before retrying with exponential backoff
                time.sleep(delay)
                
                # Increase delay for next retry (exponential backoff)
                delay *= self.backoff_factor

    def generate(self, prompt: str, audio_path: str) -> Dict[str, Any]:
        """Generate response for a single audio file"""
        try:
            # Verify audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get response from GPT-4o
            response = self._ask_with_retry(prompt, audio_path)
            
            # Try to parse JSON response if it's wrapped in code blocks
            try:
                if "```json" in response:
                    json_content = response.split("```json", 1)[1].split("```", 1)[0].strip()
                    parsed_response = json.loads(json_content)
                else:
                    parsed_response = json.loads(response)
                response_text = json.dumps(parsed_response, ensure_ascii=False)
            except:
                # If JSON parsing fails, use raw response
                response_text = response
            
            return {
                "response": response_text,
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name
            }
            
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name,
                "error": True
            }

    def batch_generate(self, prompts: List[str], audio_paths: List[str], num_workers: int = 4) -> List[Dict[str, Any]]:
        """Generate responses for multiple audio files"""
        if len(prompts) != len(audio_paths):
            raise ValueError("Number of prompts must match number of audio paths")
        
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate, prompt, audio_path) 
                      for prompt, audio_path in zip(prompts, audio_paths)]
            results = [future.result() for future in futures]
        
        return results

    def evaluate_with_schema(self, audio_path: str, prompt: str, key: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate audio using the main evaluation schema from all_prompt.py"""
        try:
            response = self._ask_with_retry(prompt, audio_path, EvaluationOutputSchema)
            return {
                "response": response,
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name,
                "error": True
            }
