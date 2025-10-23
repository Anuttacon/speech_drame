
system_prompt = "You are an expert evaluator of speech delivery and storytelling."

base_header = (
    "Based on the given information, rate the following SINGLE dimension on a 1–5 scale (1 = poor, 5 = excellent).\n"
    "Return ONLY a number between 1 and 5.\n"
)

detailed_dimension_prompts = {
    "pitch_dynamics": base_header +
        "Dimension: PitchDynamics — Variety and appropriateness of pitch contours.\n\n"
        "5: Natural, situation-appropriate melodic movement; subtle, believable inflections that reveal emotion or emphasis.\n"
        "4: Generally varied and fitting, but with minor stiffness or occasional flat spots.\n"
        "3: Neutral or limited inflection; neither distracting nor especially expressive.\n"
        "2: Forced, exaggerated, or oddly sudden shifts that feel 'performed' rather than lived.\n"
        "1: Monotone or blatantly mis-pitched for the scene (e.g., cheerfully high in a funeral).\n"
        "Example output: 4",
    
    "rhythmic_naturalness": base_header +
        "Dimension: RhythmicNaturalness — Flow, timing, absence of awkward pauses.\n\n"
        "5: Flowing, lifelike rhythm; pauses and pace reinforce intent and character habit.\n"
        "4: Mostly smooth; slight timing hiccups or mildly off beat, but still believable.\n"
        "3: Even pacing that feels read-aloud; little personality or urgency.\n"
        "2: Halting, mechanical, or oddly broken phrases; distracting pauses.\n"
        "1: Disjointed or robotic timing that destroys intelligibility or immersion.\n"
        "Example output: 3",
    
    "stress_emphasis": base_header +
        "Dimension: StressEmphasis — Clarity and correctness of syllable emphasis.\n\n"
        "5: Key syllables/words stressed perfectly; enhances meaning and emotion.\n"
        "4: Generally correct focus with a few misplaced or missing emphases.\n"
        "3: Mostly flat; only generic sentential stress.\n"
        "2: Over-emphasis or stress on function words ('of', 'the'), muddying intent.\n"
        "1: Completely flat OR every word stressed equally, causing confusion.\n"
        "Example output: 5",
    
    "emotion_accuracy": base_header +
        "Dimension: EmotionAccuracy — Does expressed emotion match intended label?\n\n"
        "5: Expressed emotion fits character & scene precisely; subtle, authentic.\n"
        "4: Appropriate but slightly understated / overstated.\n"
        "3: Emotionally neutral; not harmful, but misses clear emotional opportunity.\n"
        "2: Noticeably off (e.g., joking tone in crisis).\n"
        "1: Jarring, nonsensical emotion versus context.\n"
        "Example output: 4",
    
    "emotion_intensity": base_header +
        "Dimension: EmotionIntensity — Strength of expressed emotion.\n"
        "Note: Only score if EmotionAccuracy ≥ 3.\n\n"
        "5: Intensity calibrated perfectly to stakes and character style.\n"
        "4: Minor overshoot/undershoot but plausible.\n"
        "3: Mildly too strong/weak.\n"
        "2: Clearly over-acted or underplayed.\n"
        "1: Wildly inappropriate volume/energy.\n"
        "Example output: 2",
    
    "emotional_dynamic_range": base_header +
        "Dimension: EmotionalDynamicRange — Variation of emotion over the utterance.\n"
        "Note: Only score if EmotionIntensity ≥ 3.\n\n"
        "5: Nuanced shifts or arcs feel organic and purposeful.\n"
        "4: Contains variation, though a bit constrained or slow.\n"
        "3: Mostly steady; slight modulation.\n"
        "2: Either flat OR abrupt, unexplained jumps.\n"
        "1: Chaotic, inconsistent emotion with no narrative logic.\n"
        "Example output: 3",
    
    "voice_identity_matching": base_header +
        "Dimension: VoiceIdentityMatching — Match to speaker's known vocal timbre.\n"
        "Penalty: if explicit speaker mismatch is mentioned, answer 1.\n\n"
        "5: Voice age, timbre, energy unmistakably match profile.\n"
        "4: Acceptable with small mismatches (slightly older/younger etc.).\n"
        "3: Generic voice that could fit many characters.\n"
        "2: Noticeable contrast (e.g., deep adult voice for child).\n"
        "1: Obviously wrong identity (e.g., male for female lead).\n"
        "Example output: 5",
    
    "trait_embodiment": base_header +
        "Dimension: TraitEmbodiment — Presence of profile traits (positive OR opposite).\n"
        "• If opposite dominates, still keep trait but give low score.\n"
        "• If no traits audible, answer 1.\n\n"
        "5: Core traits audible and believable (positive or opposite direction if meaningful).\n"
        "4: Most traits present; one weak or missing.\n"
        "3: Traits faint; delivery neutral.\n"
        "2: Traits clash with profile or unclear.\n"
        "1: No discernible traits or fully off-profile.\n"
        "Example output: 2",
    
    "local_scene_fit": base_header +
        "Dimension: LocalSceneFit — Speech content and tone suit the immediate scene.\n\n"
        "5: Tone/language align perfectly with immediate objective & mood.\n"
        "4: Minor mismatch but scene still works.\n"
        "3: Neutral; neither supports nor harms the scene.\n"
        "2: Somewhat confusing or tone-deaf.\n"
        "1: Directly contradicts scene stakes.\n"
        "Example output: 4",
    
    "global_story_fit": base_header +
        "Dimension: GlobalStoryFit — Consistency with overarching story / profile.\n\n"
        "5: Consistent with long-term character arc and established habits.\n"
        "4: Slightly atypical but plausible growth.\n"
        "3: Generic filler; little personality reference.\n"
        "2: Conflicts with known backstory.\n"
        "1: Breaks established identity entirely.\n"
        "Example output: 5",
}

# Original simple dimension prompts (keeping for backward compatibility)
dimension_prompts = {
    "pitch_dynamics": base_header +
        "Dimension: PitchDynamics — Variety and appropriateness of pitch contours.\n"
        "Example output: 4",
    
    "rhythmic_naturalness": base_header +
        "Dimension: RhythmicNaturalness — Flow, timing, absence of awkward pauses.\n"
        "Example output: 3",
    
    "stress_emphasis": base_header +
        "Dimension: StressEmphasis — Clarity and correctness of syllable emphasis.\n"
        "Example output: 5",
    
    "emotion_accuracy": base_header +
        "Dimension: EmotionAccuracy — Does expressed emotion match intended label?\n"
        "Example output: 4",
    
    "emotion_intensity": base_header +
        "Dimension: EmotionIntensity — Strength of expressed emotion.\n"
        "Example output: 2",
    
    "emotional_dynamic_range": base_header +
        "Dimension: EmotionalDynamicRange — Variation of emotion over the utterance.\n"
        "Example output: 3",
    
    "voice_identity_matching": base_header +
        "Dimension: VoiceIdentityMatching — Match to speaker's known vocal timbre.\n"
        "Penalty: if explicit speaker mismatch is mentioned, answer 1.\n"
        "Example output: 5",
    
    "trait_embodiment": base_header +
        "Dimension: TraitEmbodiment — Presence of profile traits (positive OR opposite).\n"
        "• If opposite dominates, still keep trait but give low score.\n"
        "• If no traits audible, answer 1.\n"
        "Example output: 2",
    
    "local_scene_fit": base_header +
        "Dimension: LocalSceneFit — Speech content and tone suit the immediate scene.\n"
        "Example output: 4",
    
    "global_story_fit": base_header +
        "Dimension: GlobalStoryFit — Consistency with overarching story / profile.\n"
        "Example output: 5",
}

general_prompt_detailed = """**WHAT TO DO**  
1. **Read first:** Global Profile ➜ Local Scene.  
2. **Play** the audio **while skimming** the transcript (transcript may be imperfect).  
3. **Focus only on the target speaker.**  
   • Ignore background SFX / music / other voices except where they influence the target character’s timing or tone.  
4. **Apply the rubric below EXACTLY** and return **only** as the example – no prose, line-breaks, or extra keys.  
5. **Scale is 1 (poor) … 5 (excellent).** Use integers only.  

**DIMENSIONS & RULES**  
For every dimension, 5 = best possible, 1 = worst. Use these anchor descriptions to choose the closest integer:

### 1. PitchDynamics  
|5|Natural, situation-appropriate melodic movement; subtle, believable inflections that reveal emotion or emphasis.|
|4|Generally varied and fitting, but with minor stiffness or occasional flat spots.|
|3|Neutral or limited inflection; neither distracting nor especially expressive.|
|2|Forced, exaggerated, or oddly sudden shifts that feel "performed" rather than lived.|
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
|2|Over-emphasis or stress on function words ("of", "the"), muddying intent.|
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

**Penalty:** If the speaker is clearly the wrong person, set VoiceIdentityMatching = 1 and TraitEmbodiment = 1 (even if other scores are higher).
**Present results in a fixed order:** PitchDynamics, RhythmicNaturalness, StressEmphasis, EmotionAccuracy, EmotionIntensity, EmotionalDynamicRange, VoiceIdentityMatching, TraitEmbodiment, LocalSceneFit, GlobalStoryFit.

OUTPUT SCHEMA 
<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>
"""

general_prompt ="""Follow the rubric **exactly** and return **only** as the example —no extra text.

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

Penalty Explicit speaker mismatch ⇒ VoiceIdentityMatching = 1 and TraitEmbodiment = 1.
Present results in a fixed order: PitchDynamics, RhythmicNaturalness, StressEmphasis, EmotionAccuracy, EmotionIntensity, EmotionalDynamicRange, VoiceIdentityMatching, TraitEmbodiment, LocalSceneFit, GlobalStoryFit.

OUTPUT SCHEMA 
<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>
"""

caption_prompt = """In addition to the above rubric, please also provide the audio caption in <think> </think> before the answer."""
emotion_prompt = """In addition to the above rubric, please also provide the emotion caption in <think> </think> before the answer."""
