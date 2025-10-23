# Copyright 2025 Jiatong Shi (Anuttacon)
import json

system_prompt = "You are an expert evaluator of speech delivery and storytelling."

base_header = (
    "Based on the given information, rate the following SINGLE dimension on a 1–5 scale (1 = poor, 5 = excellent).\n"
    "Return ONLY a number between 1 and 5.\n"
)

# Detailed dimension prompts with 5-point scale descriptions
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
    
    "semantic_matchness": base_header +
        "Dimension: SemanticMatchness — Spoken content perfectly advances the scene goal.\n\n"
        "5: Spoken content perfectly advances the scene goal or emotion.\n"
        "4: Minor nuance mismatch but still scene-appropriate.\n"
        "3: Neutral filler that neither helps nor contradicts the scene.\n"
        "2: Partially conflicts with or confuses the scene context; requires a stretch explanation.\n"
        "1: Directly contradicts or derails the scene’s intent; no reasonable bridge.\n"
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
    
    "semantic_matchness": base_header +
        "Dimension: SemanticMatchness — Spoken content perfectly advances the scene goal.\n"
        "Example output: 5",
}

def get_all_dimension_keys() -> list[str]:
    return list(dimension_prompts.keys())

def get_dimension_prompt(key: str, detailed: bool = False) -> str:
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

def get_all_detailed_prompts() -> dict[str, str]:
    """Get all detailed dimension prompts."""
    return detailed_dimension_prompts.copy()

def get_all_simple_prompts() -> dict[str, str]:
    """Get all simple dimension prompts."""
    return dimension_prompts.copy()