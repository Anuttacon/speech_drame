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

general_prompt ="""Follow the rubric **exactly** and return **only** a minified JSON object
(valid for `json.loads`)—no extra text.

RUBRIC (all 10 dimensions, 1 = poor … 5 = excellent)
• PitchDynamics  Variety & appropriateness of pitch contours.  
• RhythmicNaturalness  Flow, timing, absence of awkward pauses.  
• StressEmphasis  Clarity & correctness of syllable/word emphasis.  
• EmotionAccuracy  Does expressed emotion match intended label?  
• EmotionIntensity  Strength of expressed emotion.  
• EmotionalDynamicRange  Variation of emotion over the utterance.  
• VoiceIdentityMatching  Matches speaker’s known vocal identity / timbre.  
• TraitEmbodiment  Presence (positive **or opposite**) of profile traits.  
  – Keep a trait if audible either way; opposite ⇒ low score.  
  – If none remain, set `"traits":"N/A"` and TraitEmbodiment = 1.  
• LocalSceneFit  Speech content & tone suit the immediate scene.  
• GlobalStoryFit  Consistent with overarching story / speaker profile.
• SemanticMatchness  Spoken content perfectly advances the scene goal.

Penalty Explicit speaker mismatch ⇒ VoiceIdentityMatching = 1 and TraitEmbodiment = 1.

OUTPUT SCHEMA  
```json
{
  "pitch_dynamics": 1.0-5.0,
  "rhythmic_naturalness": 1.0-5.0,
  "stress_emphasis": 1.0-5.0,
  "emotion_accuracy": 1.0-5.0,
  "emotion_intensity": 1.0-5.0,
  "emotional_dynamic_range": 1.0-5.0,
  "voice_identity_matching": 1.0-5.0,
  "trait_embodiment": 1.0-5.0,
  "local_scene_fit": 1.0-5.0,
  "global_story_fit": 1.0-5.0,
  "semantic_matchness": 1.0-5.0,
}```"""

general_prompt_detailed = """**WHAT TO DO**  
1. **Read first:** Global Profile ➜ Local Scene.  
2. **Play** the audio **while skimming** the transcript (transcript may be imperfect).  
3. **Focus only on the target speaker.**  
   • Ignore background SFX / music / other voices except where they influence the target character’s timing or tone.  
4. **Apply the rubric below EXACTLY** and return **only** a **minified JSON** object – no prose, line-breaks, or extra keys (valid for `json.loads`).  
5. **Scale is 1 (poor) … 5 (excellent).** Use integers only.  

**DIMENSIONS & RULES**  
For every dimension, 5 = best possible, 1 = worst. Use these anchor descriptions to choose the closest integer:

### 1. PitchDynamics  
|5|Natural, situation-appropriate melodic movement; subtle, believable inflections that reveal emotion or emphasis.|
|4|Generally varied and fitting, but with minor stiffness or occasional flat spots.|
|3|Neutral or limited inflection; neither distracting nor especially expressive.|
|2|Forced, exaggerated, or oddly sudden shifts that feel “performed” rather than lived.|
|1|Monotone or blatantly mis-pitched for the scene (e.g., cheerfully high in a funeral).|

### 2. RhythmicNaturalness  
|5|Flowing, lifelike rhythm; pauses and pace reinforce intent and character habit.|
|4|Mostly smooth; slight timing hiccups or mildly off beat, but still believable.|
|3|Even pacing that feels read-aloud; little personality or urgency.|
|2|Halting, mechanical, or oddly broken phrases; distracting pauses.|
|1|Disjointed or robotic timing that destroys intelligibility or immersion.|

### 3. StressEmphasis  
|5|Key syllables/words stressed perfectly; enhances meaning and emotion.|
|4|Generally correct focus with a few misplaced or missing emphases.|
|3|Mostly flat; only generic sentential stress.|
|2|Over-emphasis or stress on function words (“of”, “the”), muddying intent.|
|1|Completely flat OR every word stressed equally, causing confusion.|

### 4. EmotionAccuracy  
|5|Expressed emotion fits character & scene precisely; subtle, authentic.|
|4|Appropriate but slightly understated / overstated.|
|3|Emotionally neutral; not harmful, but misses clear emotional opportunity.|
|2|Noticeably off (e.g., joking tone in crisis).|
|1|Jarring, nonsensical emotion versus context.|

### 5. EmotionIntensity  *(only score if 4 ≥ 3)*  
|5|Intensity calibrated perfectly to stakes and character style.|
|4|Minor overshoot/undershoot but plausible.|
|3|Mildly too strong/weak.|
|2|Clearly over-acted or underplayed.|
|1|Wildly inappropriate volume/energy.|

### 6. EmotionalDynamicRange  *(only score if 5 ≥ 3)*  
|5|Nuanced shifts or arcs feel organic and purposeful.|
|4|Contains variation, though a bit constrained or slow.|
|3|Mostly steady; slight modulation.|
|2|Either flat OR abrupt, unexplained jumps.|
|1|Chaotic, inconsistent emotion with no narrative logic.|

### 7. VoiceIdentityMatching  
|5|Voice age, timbre, energy unmistakably match profile.|
|4|Acceptable with small mismatches (slightly older/younger etc.).|
|3|Generic voice that could fit many characters.|
|2|Noticeable contrast (e.g., deep adult voice for child).|
|1|Obviously wrong identity (e.g., male for female lead).|

### 8. TraitEmbodiment  
|5|Core traits audible and believable (positive **or opposite** direction if meaningful).|
|4|Most traits present; one weak or missing.|
|3|Traits faint; delivery neutral.|
|2|Traits clash with profile or unclear.|
|1|No discernible traits or fully off-profile.|

### 9. LocalSceneFit  
|5|Tone/language align perfectly with immediate objective & mood.|
|4|Minor mismatch but scene still works.|
|3|Neutral; neither supports nor harms the scene.|
|2|Somewhat confusing or tone-deaf.|
|1|Directly contradicts scene stakes.|

### 10. GlobalStoryFit  
|5|Consistent with long-term character arc and established habits.|
|4|Slightly atypical but plausible growth.|
|3|Generic filler; little personality reference.|
|2|Conflicts with known backstory.|
|1|Breaks established identity entirely.|

### 11. SemanticMatchness   
|5|Spoken content perfectly advances the scene goal or emotion.|
|4|Minor nuance mismatch but still scene-appropriate.|
|3|Neutral filler that neither helps nor contradicts the scene.|
|2|Partially conflicts with or confuses the scene context; requires a stretch explanation.|
|1|Directly contradicts or derails the scene’s intent; no reasonable bridge.|

**Penalty:** If the speaker is clearly the wrong person, set VoiceIdentityMatching = 1 and TraitEmbodiment = 1 (even if other scores are higher).

OUTPUT SCHEMA  
```json
{
  "pitch_dynamics": 1.0-5.0,
  "rhythmic_naturalness": 1.0-5.0,
  "stress_emphasis": 1.0-5.0,
  "emotion_accuracy": 1.0-5.0,
  "emotion_intensity": 1.0-5.0,
  "emotional_dynamic_range": 1.0-5.0,
  "voice_identity_matching": 1.0-5.0,
  "trait_embodiment": 1.0-5.0,
  "local_scene_fit": 1.0-5.0,
  "global_story_fit": 1.0-5.0,
  "semantic_matchness": 1.0-5.0,
}```"""

def get_user_prompt(user_data: dict) -> tuple[str, str]:
    """
    Set the user prompt for the model.
    """
    global_profile = user_data["char_profile"]
    local_story = user_data["local_scene"]
    speech_path = user_data["wav_path"]
    available_traits = user_data["char_style"]

    user_prompt = (
        f"Global Profile: {global_profile}, "
        f"Local Scene: {local_story}, "
        f"Available Traits: {available_traits}"
    )

    return user_prompt, speech_path

def get_system_prompt() -> str:
    return system_prompt

def get_rubric_prompt(detailed=False) -> str:
    return general_prompt



