# Copyright 2025 Jiatong Shi (Anuttacon)
from pydantic import BaseModel, Field
from typing import Optional

class EvaluationOutputSchema(BaseModel):
    """Schema for evaluation output based on the 10 dimensions from all_prompt.py"""
    
    pitch_dynamics: int = Field(..., ge=1, le=5, description="Variety & appropriateness of pitch contours")
    rhythmic_naturalness: int = Field(..., ge=1, le=5, description="Flow, timing, absence of awkward pauses")
    stress_emphasis: int = Field(..., ge=1, le=5, description="Clarity & correctness of syllable/word emphasis")
    emotion_accuracy: int = Field(..., ge=1, le=5, description="Does expressed emotion match intended label?")
    emotion_intensity: int = Field(..., ge=1, le=5, description="Strength of expressed emotion")
    emotional_dynamic_range: int = Field(..., ge=1, le=5, description="Variation of emotion over the utterance")
    voice_identity_matching: int = Field(..., ge=1, le=5, description="Matches speaker's known vocal identity / timbre")
    trait_embodiment: int = Field(..., ge=1, le=5, description="Presence (positive or opposite) of profile traits")
    local_scene_fit: int = Field(..., ge=1, le=5, description="Speech content & tone suit the immediate scene")
    global_story_fit: int = Field(..., ge=1, le=5, description="Consistent with overarching story / speaker profile")
    semantic_matchness: int = Field(..., ge=1, le=5, description="Spoken content perfectly advances the scene goal")
    
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class ArchetypeOutputSchema(BaseModel):
    """Schema for archetype output"""
    content_pass: bool = Field(..., description="If rejected, set to False, otherwise set to True")
    audio_quality: int = Field(..., ge=1, le=5, description="Artifacts/glitches only")
    human_likeness: int = Field(..., ge=1, le=5, description="Context-free naturalness")
    appropriateness: int = Field(..., ge=1, le=5, description="Context-dependent tonal fit to the prompt's role & scene")


class PitchDynamicsOutputSchema(BaseModel):
    """Schema for pitch dynamics output"""
    
    pitch_dynamics: int = Field(..., ge=1, le=5, description="Variety & appropriateness of pitch contours")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class RhythmicNaturalnessOutputSchema(BaseModel):
    """Schema for rhythmic naturalness output"""
    
    rhythmic_naturalness: int = Field(..., ge=1, le=5, description="Flow, timing, absence of awkward pauses")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class StressEmphasisOutputSchema(BaseModel):
    """Schema for stress emphasis output"""
    
    stress_emphasis: int = Field(..., ge=1, le=5, description="Clarity & correctness of syllable/word emphasis")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class EmotionAccuracyOutputSchema(BaseModel):
    """Schema for emotion accuracy output"""
    
    emotion_accuracy: int = Field(..., ge=1, le=5, description="Does expressed emotion match intended label?")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class EmotionIntensityOutputSchema(BaseModel):
    """Schema for emotion intensity output"""

    emotion_intensity: int = Field(..., ge=1, le=5, description="Strength of expressed emotion")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class EmotionalDynamicRangeOutputSchema(BaseModel):
    """Schema for emotional dynamic range output"""

    emotional_dynamic_range: int = Field(..., ge=1, le=5, description="Variation of emotion over the utterance")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class VoiceIdentityMatchingOutputSchema(BaseModel):
    """Schema for voice identity matching output"""

    voice_identity_matching: int = Field(..., ge=1, le=5, description="Matches speaker's known vocal identity / timbre")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class TraitEmbodimentOutputSchema(BaseModel):
    """Schema for trait embodiment output"""

    trait_embodiment: int = Field(..., ge=1, le=5, description="Presence (positive or opposite) of profile traits")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class LocalSceneFitOutputSchema(BaseModel):
    """Schema for local scene fit output"""

    local_scene_fit: int = Field(..., ge=1, le=5, description="Speech content & tone suit the immediate scene")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class GlobalStoryFitOutputSchema(BaseModel):
    """Schema for global story fit output"""

    global_story_fit: int = Field(..., ge=1, le=5, description="Consistent with overarching story / speaker profile")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

class SemanticMatchnessOutputSchema(BaseModel):
    """Schema for semantic matchness output"""

    semantic_matchness: int = Field(..., ge=1, le=5, description="Spoken content perfectly advances the scene goal")
    # Optional field for traits when none remain
    traits: Optional[str] = Field(None, description="Available traits or 'N/A' if none remain")

def get_dimension_schema(key: str) -> BaseModel:
    """Get the schema for a specific dimension"""
    if key == "pitch_dynamics":
        return PitchDynamicsOutputSchema
    elif key == "rhythmic_naturalness":
        return RhythmicNaturalnessOutputSchema
    elif key == "stress_emphasis":
        return StressEmphasisOutputSchema
    elif key == "emotion_accuracy":
        return EmotionAccuracyOutputSchema
    elif key == "emotion_intensity":
        return EmotionIntensityOutputSchema
    elif key == "emotional_dynamic_range":
        return EmotionalDynamicRangeOutputSchema
    elif key == "voice_identity_matching":
        return VoiceIdentityMatchingOutputSchema
    elif key == "trait_embodiment":
        return TraitEmbodimentOutputSchema
    elif key == "local_scene_fit":
        return LocalSceneFitOutputSchema
    elif key == "global_story_fit":
        return GlobalStoryFitOutputSchema
    elif key == "semantic_matchness":
        return SemanticMatchnessOutputSchema
    else:
        raise ValueError(f"Invalid dimension key: {key}")



class UnderstandingOutputSchema(BaseModel):
    """Schema for general audio understanding output"""
    
    transcription: str = Field(..., description="Transcribed speech content")
    emotion: str = Field(..., description="Detected emotional state")
    speaker_characteristics: str = Field(..., description="Speaker voice characteristics")
    audio_quality: str = Field(..., description="Audio quality assessment")
    context: str = Field(..., description="Overall context and meaning")

class AudioQualityOutputSchema(BaseModel):
    """Schema for audio quality evaluation output"""
    
    clarity: int = Field(..., ge=1, le=5, description="Audio clarity and intelligibility")
    noise_level: int = Field(..., ge=1, le=5, description="Background noise level")
    balance: int = Field(..., ge=1, le=5, description="Audio balance and mixing")
    production_quality: int = Field(..., ge=1, le=5, description="Overall production quality")
    technical_issues: str = Field(..., description="Any technical issues or artifacts found")
    overall_score: int = Field(..., ge=1, le=5, description="Overall quality score")

class EmotionAnalysisOutputSchema(BaseModel):
    """Schema for emotion analysis output"""
    
    primary_emotion: str = Field(..., description="Primary emotional state")
    emotion_intensity: int = Field(..., ge=1, le=5, description="Intensity of the emotion")
    emotional_transitions: str = Field(..., description="Any emotional transitions detected")
    tone_mood: str = Field(..., description="Overall tone and mood")
    emotional_context: str = Field(..., description="Emotional context and reasoning")

class SpeakerCharacteristicsOutputSchema(BaseModel):
    """Schema for speaker characteristics analysis output"""
    
    voice_quality: str = Field(..., description="Voice quality and timbre")
    speaking_style: str = Field(..., description="Speaking style and mannerisms")
    accent_pronunciation: str = Field(..., description="Accent and pronunciation details")
    vocal_range: str = Field(..., description="Vocal range and pitch characteristics")
    personality_traits: str = Field(..., description="Overall speaker personality traits")

class AudioEventsOutputSchema(BaseModel):
    """Schema for audio events detection output"""
    
    speech_content: str = Field(..., description="Speech and language content")
    background_sounds: str = Field(..., description="Background sounds and ambient noise")
    music_elements: str = Field(..., description="Music or musical elements")
    sound_effects: str = Field(..., description="Sound effects or audio artifacts")
    temporal_sequence: str = Field(..., description="Temporal sequence of events") 