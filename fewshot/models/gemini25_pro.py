from .base_audiollm import BaseAudioLLM
from typing import Dict, Any, List
import os
import json
import time
from google import genai
from google.genai.types import Part, HttpOptions, Content
from concurrent.futures import ThreadPoolExecutor
from prompts.schema import (
    EvaluationOutputSchema, 
    ArchetypeOutputSchema,
    get_dimension_schema,
)
from typing import Optional

class Gemini25Pro(BaseAudioLLM):
    def __init__(self, model_name: str, model_tag: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.model_tag = model_tag
        self.system_prompt = system_prompt or "You are a helpful AI assistant that can understand and analyze audio content."
        
        # Setup Google credentials
        credential_json_path = "crediential.json"
        if os.path.exists(credential_json_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_json_path
        
        # Initialize Gemini client
        self.client = genai.Client(
            vertexai=True,
            project="gen-lang-client-0861345556",
            location="global",
            http_options=HttpOptions(api_version='v1')
        )
        
        # Retry settings
        self.max_retries = 10
        self.initial_delay = 1
        self.backoff_factor = 1.15

    def _ask_with_retry(self, question: str, audio_path: str, output_schema: Any = None) -> str:
        """Ask a question about the audio with retry logic"""
        
        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        retry_count = 0
        delay = self.initial_delay
        
        while True:
            try:
                # Prepare request config
                config = {}
                if output_schema:
                    config.update({
                        'response_mime_type': 'application/json',
                        'response_schema': output_schema,
                    })
                
                # Make API call
                response = self.client.models.generate_content(
                    model=self.model_tag,
                    contents=[
                        Part.from_text(text=question),
                        Part.from_bytes(data=audio_data, mime_type="audio/wav"),
                    ],
                    config=config
                )
                return response.text
                
            except Exception as e:
                retry_count += 1
                
                # Check if retries are exhausted
                if retry_count > self.max_retries:
                    print(f"Failed after {self.max_retries} retries. Last error: {str(e)}")
                    raise e
                    
                # Log the retry attempt
                print(f"Error during Gemini API call: {str(e)}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                
                # Wait before retrying with exponential backoff
                time.sleep(delay)
                
                # Increase delay for next retry (exponential backoff)
                delay *= self.backoff_factor

    def _wav_to_base64(self, wav_path: str) -> str:
        """Convert WAV file to base64 string"""
        import base64
        with open(wav_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')


    def _ask_with_retry_messages(self, messages: List[Dict], output_schema: Any = None) -> str:
        """Ask a question using message format with retry logic"""
        retry_count = 0
        delay = self.initial_delay
        
        while True:
            try:
                # Prepare request config
                config = {}
                if output_schema:
                    config.update({
                        'response_mime_type': 'application/json',
                        'response_schema': output_schema,
                    })
                
                # Convert messages to Gemini format
                contents = []
                for message in messages:
                    if message["role"] == "system":
                        # System messages are typically handled differently in Gemini
                        continue
                    elif message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "text":
                                contents.append(Part.from_text(text=content["text"]))
                            elif content["type"] == "input_audio":
                                # Decode base64 audio data
                                import base64
                                audio_data = base64.b64decode(content["input_audio"]["data"].split(",")[1])
                                contents.append(Part.from_bytes(data=audio_data, mime_type="audio/wav"))
                
                # Make API call
                response = self.client.models.generate_content(
                    model=self.model_tag,
                    contents=contents,
                    config=config
                )
                return response.text
                
            except Exception as e:
                retry_count += 1
                
                # Check if retries are exhausted
                if retry_count > self.max_retries:
                    print(f"Failed after {self.max_retries} retries. Last error: {str(e)}")
                    raise e
                    
                # Log the retry attempt
                print(f"Error during Gemini API call: {str(e)}. Retrying ({retry_count}/{self.max_retries}) after {delay} seconds...")
                
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
            
            # Create full prompt with system message
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            # Get response from Gemini
            response = self._ask_with_retry(full_prompt, audio_path)
            
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
        if key is None:
            output_schema = EvaluationOutputSchema
        elif key == "archetype":
            output_schema = ArchetypeOutputSchema
        else:
            output_schema = get_dimension_schema(key)

        try:
            response = self._ask_with_retry(prompt, audio_path, output_schema)
            if key is not None and key != "archetype":
                response = json.loads(response)[key]
            return {
                "response": response,
                "audio_path": audio_path,
                "prompt": prompt,
                "model_name": self.model_name
            }
        except Exception as e:
            import traceback
            print(f"Error in schema evaluation: {str(e)}")
            print(traceback.format_exc())
            raise e
        # except Exception as e:
        #     return {
        #         "response": f"Error: {str(e)}",
        #         "audio_path": audio_path,
        #         "prompt": prompt,
        #         "model_name": self.model_name,
        #         "error": True
        #     }

    def evaluate_few_shot_with_schema(self, few_shot_examples: List[Dict], current_item: Dict, rubric_prompt: str, key: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate using few-shot examples with base64 audio in single user prompt"""
        if key is None:
            output_schema = EvaluationOutputSchema
        elif key == "archetype":
            output_schema = ArchetypeOutputSchema
        else:
            output_schema = get_dimension_schema(key)

        try:
            # Build the prompt with few-shot examples
            prompt_parts = []
            
            # Start with rubric
            prompt_parts.append({
                "type": "text",
                "text": f"{rubric_prompt}\n\n"
            })
            
            # Add few-shot examples
            for i, ex in enumerate(few_shot_examples):
                prompt_parts.append({
                    "type": "text", 
                    "text": f"## **Few-shot Example {i+1}**\n\n"
                })
                
                if key == "archetype":
                    example_prompt = f"Prompt’s role & scene: {ex.get('question', '')}"
                    prompt_parts.append({
                        "type": "text",
                        "text": f"**Example Input:** {example_prompt}\n\n"
                    })
                else:
                    # Add example prompt
                    example_prompt = f"Global Profile: {ex.get('char_profile', '')}, Local Scene: {ex.get('local_scene', '')}, Available Traits: {ex.get('char_style', '')}"
                    prompt_parts.append({
                        "type": "text",
                        "text": f"**Example Input:** {example_prompt}\n\n"
                    })

                
                # Add example audio as base64
                example_audio_base64 = self._wav_to_base64(ex["wav_path"])
                prompt_parts.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64," + example_audio_base64,
                        "format": "wav"
                    }
                })
                
                # Add example output
                if 'annotation_prompt' in ex:
                    gold_scores = ex['annotation_prompt']
                    gold_json = json.dumps(gold_scores, indent=2)
                    prompt_parts.append({
                        "type": "text",
                        "text": f"\n\n**Example Output:** {gold_json}\n\n"
                    })
            
            # Add current evaluation
            prompt_parts.append({
                "type": "text",
                "text": "## **Current Evaluation**\n\n"
            })
            
            if key == "archetype":
                current_prompt = f"Prompt’s role & scene: {current_item.get('question', '')}"
            else:
                current_prompt = f"Global Profile: {current_item.get('char_profile', '')}, Local Scene: {current_item.get('local_scene', '')}, Available Traits: {current_item.get('char_style', '')}"
            prompt_parts.append({
                "type": "text",
                "text": f"**Current Input:** {current_prompt}\n\n"
            })
            
            # Add current audio as base64
            current_audio_base64 = self._wav_to_base64(current_item["wav_path"])
            prompt_parts.append({
                "type": "input_audio",
                "input_audio": {
                    "data": "data:audio/wav;base64," + current_audio_base64,
                    "format": "wav"
                }
            })
            
            prompt_parts.append({
                "type": "text",
                "text": "\n\n**Please evaluate the current audio and provide your response in the same JSON format as the examples above.**"
            })
            
            # Create single user message with all content
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt_parts
                }
            ]
            
            # Make API call
            response = self._ask_with_retry_messages(messages, output_schema)
            
            if key is not None and key != "archetype":
                response = json.loads(response)[key]
            
            return {
                "response": response,
                "audio_path": current_item["wav_path"],
                "prompt": current_prompt,
                "model_name": self.model_name
            }
        except Exception as e:
            import traceback
            print(f"Error in few-shot evaluation: {str(e)}")
            print(traceback.format_exc())
            raise e
