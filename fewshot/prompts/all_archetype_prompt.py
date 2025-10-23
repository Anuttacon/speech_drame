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

general_prompt = """Follow the rubric EXACTLY and return ONLY a minified JSON object (valid for json.loads)—no extra text.
Set content_pass to false (i.e., reject) if ANY apply:
• Clip ~1–2s (too short to judge).
• Obvious breakdown/loop/restart.
• Any multilingual word/phrase (except extremely common one-offs like “croissant”).
• Irrelevant to prompt’s role/scene.
• Instruction failure: breaks persona/scene, 4th-wall narration, multi-party narration, long placeholder scaffolding.
Otherwise: PASS.

1) If not rejected, judge performance (HOW it’s said), not wording.
2) If rejected, set all scores to 1.

RUBRIC (three dimensions, 1 = poor … 5 = excellent)
• Content-Pass: if rejected, set to false, otherwise set to true.
• AudioQuality—Artifacts/glitches only. After recording this score, pretend these issues do not exist for later dimensions.
• HumanLikeness—Context-free naturalness; ignore quality artifacts and contextual fit.
• Appropriateness—Context-dependent tonal fit to the prompt’s role & scene; judge harshly against the stereotype. If any portion before 30s is off, pick the lower score. Ignore quality & generic naturalness.

OUTPUT SCHEMA (minified JSON)
{"content_pass":true/false,"audio_quality":1-5,"human_likeness":1-5,"appropriateness":1-5}"""

# -------------------------
# DETAILED RUBRIC (V4)
# -------------------------

general_prompt_detailed = f"""Follow the rubric EXACTLY and return ONLY a minified JSON object (valid for json.loads)—no extra text.
Set content_pass to false (i.e., reject) if ANY apply:
• Clip ~1–2s (too short to judge).
• Obvious breakdown/loop/restart.
• Any multilingual word/phrase (except extremely common one-offs like “croissant”).
• Irrelevant to prompt’s role/scene.
• Instruction failure: breaks persona/scene, 4th-wall narration, multi-party narration, long placeholder scaffolding.
Otherwise: PASS.

**WHAT TO DO**
1) Apply the rubrics below strictly. If torn between two scores, choose the LOWER one.
2) Return ONLY a minified JSON object (valid for json.loads). Integers 1–5. No extra text.
3) If not rejected, judge performance (HOW it’s said), not wording.
4) If rejected, set all scores to 1.

---

### 0) Content-Pass
If rejected, set to false, otherwise set to true.

### 1) AudioQuality (Context-Free)
Focus ONLY on digital/synthetic artifacts (noise, hiss, metallic timbre, clipping, glitchy transitions, pitch collapse, static).
After you record this score, PRETEND these issues do not exist for later dimensions.

Scale:
- 5 (Excellent): Clean/clear; no noticeable artifacts.
- 4 (Good): Minor imperfections (slight hiss/sharpness) under close listen.
- 3 (Fair): Noticeable artifacts (buzz/echo/synthetic texture) that impact clarity.
- 2 (Poor): Frequent glitches/clipping/metallic resonance/harsh transitions.
- 1 (Very Poor): Dominated by artifacts (static/crackling/synthesis failure/severe clipping or pitch collapse).

### 2) HumanLikeness (Context-Free)
How human-like is the delivery? Ignore AudioQuality issues (already captured) and ignore contextual fit.

Scale:
- 5 (Definitely Human): Indistinguishable; rich intonation, natural real-time pacing, consistent identity; organic breaths/pauses.
- 4 (Most Likely Human): Generally convincing; minor stiffness or overly even timing/pauses.
- 3 (Could be Human or AI): Mixed traits; some mechanical pauses/inflection, other parts human-like.
- 2 (Mostly Likely AI): Clearly artificial; choppy rhythm, flat/illogical inflections, awkward or forced breaths.
- 1 (Definitely AI): Rigid read-aloud monotony; no natural rhythm/pauses; entirely machine-like.

### 3) Appropriateness (Context-Dependent)
Tonal fit to the prompt’s role & scene (stereotype-based). If ANY portion before 30s is ill-fitting, choose the LOWER score.
Ignore AudioQuality and generic naturalness here—judge only contextual tone/demeanor/status/urgency/emotion.

Scale:
- 5 (Completely Appropriate): Fits both role AND scene perfectly; fully aligned with emotion, social dynamics, status, and events; immersive and consistent.
- 4 (Mostly Appropriate): Strong match with minor flaws (e.g., slight intensity/timing/subtlety issues) that do not break alignment.
- 3 (Adequately Appropriate): Broad fit to role/scene but lacks depth or shows mild inconsistencies in key elements.
- 2 (Slightly Appropriate): Only role OR scene fit is acceptable; superficial/partial alignment; missing/mismatched cues.
- 1 (Completely Inappropriate): Fails role AND scene; tone/status/urgency clearly mismatched.

**OUTPUT SCHEMA (minified JSON)**
{{"content_pass":true/false,"audio_quality":1-5,"human_likeness":1-5,"appropriateness":1-5}}
"""

# -------------------------
# Helpers & API
# -------------------------

def get_archetype_user_prompt(user_data: dict) -> tuple[str, str]:
    """
    Set the user prompt for the model.
    """
    question = user_data["question"]
    speech_path = user_data["wav_path"]

    user_prompt = f"Prompt’s role & scene: {question}"
    return user_prompt, speech_path

def get_archetype_system_prompt() -> str:
    return system_prompt

def get_archetype_rubric_prompt(detailed: bool = False) -> str:
    return general_prompt_detailed if detailed else general_prompt
