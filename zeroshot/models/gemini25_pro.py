from .base_audiollm import BaseAudioLLM
from typing import Dict, Any, List
import os
import json
import time
from google import genai
from google.genai.types import Part, HttpOptions
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
        
        # Use the caller's credentials/project; never a developer-specific account.
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true"}
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if use_vertex or (os.getenv("GOOGLE_CLOUD_PROJECT") and not key):
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError("Vertex requires GOOGLE_CLOUD_PROJECT and configured application default credentials")
            self.client = genai.Client(vertexai=True, project=project,
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
                http_options=HttpOptions(api_version="v1"))
        elif key:
            self.client = genai.Client(vertexai=False, api_key=key)
        else:
            raise ValueError("Set GEMINI_API_KEY/GOOGLE_API_KEY, or GOOGLE_CLOUD_PROJECT with Vertex ADC")

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