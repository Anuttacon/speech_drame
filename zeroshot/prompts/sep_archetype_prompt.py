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
import json

system_prompt = "You are an expert evaluator of speech delivery and storytelling."

base_header = (
    "Based on the given information, rate the following SINGLE dimension on a 1–5 scale (1 = poor, 5 = excellent).\n"
    "Return ONLY a number between 1 and 5.\n"
)

# Detailed dimension prompts with 5-point scale descriptions
detailed_dimension_prompts = {
    "audio_quality": base_header +
        "Dimension: AudioQuality — Digital/synthetic artifacts.\n\n"
        "5: Clean/clear; no noticeable artifacts.\n"
        "4: Minor imperfections (slight hiss/sharpness) under close listen.\n"
        "3: Noticeable artifacts (buzz/echo/synthetic texture) that impact clarity.\n"
        "2: Frequent glitches/clipping/metallic resonance/harsh transitions.\n"
        "1: Dominated by artifacts (static/crackling/synthesis failure/severe clipping or pitch collapse).\n"
        "Example output: 4",
    
    "human_likeness": base_header +
        "Dimension: HumanLikeness — Context-free naturalness.\n\n"
        "5: Indistinguishable; rich intonation, natural real-time pacing, consistent identity; organic breaths/pauses.\n"
        "4: Generally convincing; minor stiffness or overly even timing/pauses.\n"
        "3: Mixed traits; some mechanical pauses/inflection, other parts human-like.\n"
        "2: Clearly artificial; choppy rhythm, flat/illogical inflections, awkward or forced breaths.\n"
        "1: Rigid read-aloud monotony; no natural rhythm/pauses; entirely machine-like.\n"
        "Example output: 4",
    
    "appropriateness": base_header +
        "Dimension: Appropriateness — Context-dependent tonal fit to the prompt’s role & scene.\n\n"
        "5: Fits both role AND scene perfectly; fully aligned with emotion, social dynamics, status, and events; immersive and consistent.\n"
        "4: Strong match with minor flaws (e.g., slight intensity/timing/subtlety issues) that do not break alignment.\n"
        "3: Broad fit to role/scene but lacks depth or shows mild inconsistencies in key elements.\n"
        "2: Only role OR scene fit is acceptable; superficial/partial alignment; missing/mismatched cues.\n"
        "1: Fails role AND scene; tone/status/urgency clearly mismatched.\n"
        "Example output: 4",

    "content_pass": base_header + 
        "Dimension: ContentPass — If rejected, set to false, otherwise set to true.\n\n"
        "0: Reject\n"
        "1: Pass\n"
        "Example output: 1",
}

# Original simple dimension prompts (keeping for backward compatibility)
dimension_prompts = {
    "audio_quality": base_header +
        "Dimension: AudioQuality — Digital/synthetic artifacts.\n"
        "Example output: 4",
    
    "human_likeness": base_header +
        "Dimension: HumanLikeness — Context-free naturalness.\n"
        "Example output: 3",
    
    "appropriateness": base_header +
        "Dimension: Appropriateness — Context-dependent tonal fit to the prompt’s role & scene.\n"
        "Example output: 5",
    
    "content_pass": base_header +
        "Dimension: ContentPass — If rejected, set to false, otherwise set to true.\n"
        "Example output: 0",
}

def get_all_archetype_dimension_keys() -> list[str]:
    return list(dimension_prompts.keys())

def get_archetype_dimension_prompt(key: str, detailed: bool = False) -> str:
    """
    Get dimension prompt for a specific key.
    
    Args:
        key: The dimension key
        detailed: If True, return detailed prompt with 5-point scale descriptions
    
    Returns:
        The prompt string for the specified dimension
    """
    if detailed:
        return detailed_dimension_prompts.get(key, dimension_prompts.get(key, ""))
    return dimension_prompts.get(key, "")

def get_all_archetype_detailed_prompts() -> dict[str, str]:
    """Get all detailed dimension prompts."""
    return detailed_dimension_prompts.copy()

def get_all_archetype_simple_prompts() -> dict[str, str]:
    """Get all simple dimension prompts."""
    return dimension_prompts.copy()