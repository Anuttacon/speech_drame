#!/usr/bin/env python3
"""
Script to generate 100 variations of speech evaluation prompts.
Creates a prompt_variations.json file with diverse phrasings and structures.
"""

import json
import random
from typing import Dict, List

def generate_base_header_variations() -> List[str]:
    """Generate variations of the base header instruction."""
    return [
        "Based on the given information, rate the following SINGLE dimension on a 1–5 scale (1 = poor, 5 = excellent).\nReturn only the number wrapped in <answer> tags.\n",
        "Evaluate the SINGLE dimension below using a 1-5 rating scale (1 = poor, 5 = excellent).\nProvide ONLY the numeric score wrapped in <answer> tags.\n",
        "Score the following dimension on a scale of 1 to 5, where 1 is poor and 5 is excellent.\nReturn only the number wrapped in <answer> tags.\n",
        "Rate this SINGLE aspect from 1 (poor) to 5 (excellent) based on the provided context.\nOutput only the numerical rating wrapped in <answer> tags.\n",
        "Using the scale 1=poor to 5=excellent, evaluate the dimension specified below.\nRespond with only a single number wrapped in <answer> tags.\n",
        "Assess the following dimension on a 1-5 scale (1=poor, 5=excellent).\nReturn the score only wrapped in <answer> tags.\n",
        "Please rate the SINGLE dimension described below from 1 (worst) to 5 (best).\nProvide only the numeric value wrapped in <answer> tags.\n",
        "Evaluate this specific dimension using a 1-5 rating where 1 indicates poor performance and 5 indicates excellent performance.\nReturn only the number wrapped in <answer> tags.\n",
        "Score the dimension below on a scale from 1 (inadequate) to 5 (outstanding).\nOutput just the rating wrapped in <answer> tags.\n",
        "Rate the following SINGLE aspect from 1-5 based on the given information (1=poor, 5=excellent).\nProvide only the numerical score wrapped in <answer> tags.\n"
    ]

def generate_dimension_variations() -> Dict[str, List[Dict[str, str]]]:
    """Generate variations for each dimension with different phrasings and emphasis."""
    
    variations = {
        "pitch_dynamics": [
            {
                "name": "PitchDynamics",
                "description": "Variety and appropriateness of pitch contours.",
                "scale_5": "Natural, situation-appropriate melodic movement; subtle, believable inflections that reveal emotion or emphasis.",
                "scale_4": "Generally varied and fitting, but with minor stiffness or occasional flat spots.",
                "scale_3": "Neutral or limited inflection; neither distracting nor especially expressive.",
                "scale_2": "Forced, exaggerated, or oddly sudden shifts that feel 'performed' rather than lived.",
                "scale_1": "Monotone or blatantly mis-pitched for the scene (e.g., cheerfully high in a funeral)."
            },
            {
                "name": "PitchDynamics",
                "description": "Range and suitability of vocal pitch patterns.",
                "scale_5": "Organic, contextually perfect melodic flow; nuanced inflections that authentically convey emotion.",
                "scale_4": "Well-varied and appropriate, with slight rigidity or minor monotonous segments.",
                "scale_3": "Moderate inflection; adequately functional without being particularly engaging.",
                "scale_2": "Artificial, overstated, or jarringly abrupt changes that seem rehearsed rather than authentic.",
                "scale_1": "Completely flat delivery or wildly inappropriate pitch for the context."
            },
            {
                "name": "PitchDynamics",
                "description": "Quality and appropriateness of pitch movement patterns.",
                "scale_5": "Seamless, scene-fitting tonal variations; delicate, credible modulations that enhance meaning.",
                "scale_4": "Mostly dynamic and suitable, though somewhat constrained or occasionally lifeless.",
                "scale_3": "Basic inflection present; neither particularly helpful nor harmful to the delivery.",
                "scale_2": "Contrived, excessive, or strangely timed pitch shifts that distract from authenticity.",
                "scale_1": "Robotic monotone or completely mismatched pitch for the emotional context."
            }
        ],
        
        "rhythmic_naturalness": [
            {
                "name": "RhythmicNaturalness",
                "description": "Flow, timing, absence of awkward pauses.",
                "scale_5": "Flowing, lifelike rhythm; pauses and pace reinforce intent and character habit.",
                "scale_4": "Mostly smooth; slight timing hiccups or mildly off beat, but still believable.",
                "scale_3": "Even pacing that feels read-aloud; little personality or urgency.",
                "scale_2": "Halting, mechanical, or oddly broken phrases; distracting pauses.",
                "scale_1": "Disjointed or robotic timing that destroys intelligibility or immersion."
            },
            {
                "name": "RhythmicNaturalness",
                "description": "Natural timing and rhythm in speech delivery.",
                "scale_5": "Effortless, authentic cadence; strategic pauses that support character and narrative intent.",
                "scale_4": "Generally fluid; minor rhythm disruptions or slightly unnatural beats, yet remains convincing.",
                "scale_3": "Consistent tempo resembling reading aloud; lacks distinctive personality or emotional drive.",
                "scale_2": "Choppy, rigid, or strangely fragmented delivery; intrusive pausing patterns.",
                "scale_1": "Completely broken timing that impairs comprehension or narrative engagement."
            },
            {
                "name": "RhythmicNaturalness",
                "description": "Naturalness of rhythm, pacing, and pause placement.",
                "scale_5": "Organic, character-consistent tempo; breath and pause timing enhances storytelling.",
                "scale_4": "Largely natural flow with occasional timing awkwardness, but maintains believability.",
                "scale_3": "Steady, predictable pace typical of reading; minimal rhythmic personality.",
                "scale_2": "Stiff, automated, or peculiarly segmented phrasing; noticeable timing issues.",
                "scale_1": "Severely fragmented or mechanical rhythm that breaks narrative immersion."
            }
        ],
        
        "stress_emphasis": [
            {
                "name": "StressEmphasis",
                "description": "Clarity and correctness of syllable emphasis.",
                "scale_5": "Key syllables/words stressed perfectly; enhances meaning and emotion.",
                "scale_4": "Generally correct focus with a few misplaced or missing emphases.",
                "scale_3": "Mostly flat; only generic sentential stress.",
                "scale_2": "Over-emphasis or stress on function words ('of', 'the'), muddying intent.",
                "scale_1": "Completely flat OR every word stressed equally, causing confusion."
            },
            {
                "name": "WordEmphasis",
                "description": "Precision and effectiveness of lexical stress patterns.",
                "scale_5": "Optimal syllable/word highlighting that amplifies meaning and emotional impact.",
                "scale_4": "Mostly accurate emphasis placement with occasional minor misallocations.",
                "scale_3": "Basic sentence-level stress; lacks nuanced emphasis variation.",
                "scale_2": "Inappropriate stress on articles or prepositions, obscuring intended meaning.",
                "scale_1": "No stress variation OR uniform emphasis that creates semantic confusion."
            },
            {
                "name": "AccentPlacement",
                "description": "Appropriateness and clarity of stress distribution.",
                "scale_5": "Flawless emphasis on critical elements; stress patterns support comprehension and feeling.",
                "scale_4": "Sound emphasis choices with minor stress omissions or slight mispositioning.",
                "scale_3": "Standard stress patterns without special emphasis; functionally adequate.",
                "scale_2": "Misplaced stress on minor words or conjunctions, creating unclear communication.",
                "scale_1": "Absent stress variation OR every element equally emphasized, hindering understanding."
            }
        ],
        
        "emotion_accuracy": [
            {
                "name": "EmotionAccuracy",
                "description": "Does expressed emotion match intended label?",
                "scale_5": "Expressed emotion fits character & scene precisely; subtle, authentic.",
                "scale_4": "Appropriate but slightly understated / overstated.",
                "scale_3": "Emotionally neutral; not harmful, but misses clear emotional opportunity.",
                "scale_2": "Noticeably off (e.g., joking tone in crisis).",
                "scale_1": "Jarring, nonsensical emotion versus context."
            },
            {
                "name": "EmotionAccuracy",
                "description": "Correspondence between intended and expressed emotional content.",
                "scale_5": "Perfect emotional match for character and situation; refined, believable execution.",
                "scale_4": "Suitable emotion with minor intensity variations or slight tonal mismatches.",
                "scale_3": "Emotionally bland; safe but fails to capitalize on clear emotional moments.",
                "scale_2": "Clearly inappropriate emotional tone for the context (e.g., levity during tragedy).",
                "scale_1": "Completely wrong emotional expression that contradicts the scene's requirements."
            },
            {
                "name": "EmotionAccuracy",
                "description": "How well the expressed feeling aligns with the intended emotion.",
                "scale_5": "Seamless emotional authenticity; expressed affect perfectly serves character and narrative.",
                "scale_4": "Generally accurate emotion, though marginally too weak or strong for the moment.",
                "scale_3": "Emotionally vacant; neither enhances nor detracts from obvious emotional cues.",
                "scale_2": "Tangibly mismatched emotion that confuses the scene's emotional landscape.",
                "scale_1": "Bizarrely inappropriate affect that undermines narrative coherence."
            }
        ],
        
        "emotion_intensity": [
            {
                "name": "EmotionIntensity",
                "description": "Strength of expressed emotion.",
                "scale_5": "Intensity calibrated perfectly to stakes and character style.",
                "scale_4": "Minor overshoot/undershoot but plausible.",
                "scale_3": "Mildly too strong/weak.",
                "scale_2": "Clearly over-acted or underplayed.",
                "scale_1": "Wildly inappropriate volume/energy."
            },
            {
                "name": "EmotionIntensity",
                "description": "Level and appropriateness of emotional force in delivery.",
                "scale_5": "Optimal emotional strength perfectly matched to situation gravity and character traits.",
                "scale_4": "Slight intensity miscalibration, but remains within believable character bounds.",
                "scale_3": "Somewhat excessive or insufficient emotional force for the context.",
                "scale_2": "Obviously overplayed melodrama or conspicuous emotional underperformance.",
                "scale_1": "Absurdly extreme energy levels completely divorced from situational requirements."
            },
            {
                "name": "EmotionIntensity",
                "description": "Power and suitability of emotional expression intensity.",
                "scale_5": "Flawlessly modulated emotional power that serves both character authenticity and scene needs.",
                "scale_4": "Generally appropriate intensity with minor deviation from optimal emotional pitch.",
                "scale_3": "Moderately misaligned emotional strength that doesn't quite fit the moment.",
                "scale_2": "Significantly over-dramatized or noticeably emotionally muted performance.",
                "scale_1": "Completely inappropriate emotional volume that destroys scene credibility."
            }
        ],
        
        "emotional_dynamic_range": [
            {
                "name": "EmotionalDynamicRange",
                "description": "Variation of emotion over the utterance.",
                "scale_5": "Nuanced shifts or arcs feel organic and purposeful.",
                "scale_4": "Contains variation, though a bit constrained or slow.",
                "scale_3": "Mostly steady; slight modulation.",
                "scale_2": "Either flat OR abrupt, unexplained jumps.",
                "scale_1": "Chaotic, inconsistent emotion with no narrative logic."
            },
            {
                "name": "EmotionalDynamicRange",
                "description": "Quality and naturalness of emotional change throughout the speech.",
                "scale_5": "Sophisticated emotional progression that feels authentic and serves the narrative arc.",
                "scale_4": "Good emotional variation, though somewhat limited in scope or pacing.",
                "scale_3": "Fairly consistent emotional tone with minimal but present modulation.",
                "scale_2": "Either monotonous delivery OR jarring, unmotivated emotional transitions.",
                "scale_1": "Incoherent emotional chaos that lacks any discernible pattern or purpose."
            },
            {
                "name": "EmotionalDynamicRange",
                "description": "Range and naturalness of emotional movement within the delivery.",
                "scale_5": "Elegant emotional choreography; changes feel inevitable and character-driven.",
                "scale_4": "Solid emotional range with minor restrictions in fluidity or development speed.",
                "scale_3": "Predominantly stable emotion with subtle shifts that add mild interest.",
                "scale_2": "Lacks emotional movement OR features sudden, inexplicable emotional leaps.",
                "scale_1": "Random, contradictory emotional states that undermine narrative coherence."
            }
        ],
        
        "voice_identity_matching": [
            {
                "name": "VoiceIdentityMatching",
                "description": "Match to speaker's known vocal timbre.",
                "scale_5": "Voice age, timbre, energy unmistakably match profile.",
                "scale_4": "Acceptable with small mismatches (slightly older/younger etc.).",
                "scale_3": "Generic voice that could fit many characters.",
                "scale_2": "Noticeable contrast (e.g., deep adult voice for child).",
                "scale_1": "Obviously wrong identity (e.g., male for female lead)."
            },
            {
                "name": "VoiceIdentityMatching",
                "description": "Alignment between voice characteristics and established speaker identity.",
                "scale_5": "Vocal qualities perfectly embody the speaker's age, personality, and physical characteristics.",
                "scale_4": "Generally consistent voice with minor deviations in age or energy representation.",
                "scale_3": "Neutral vocal identity that doesn't contradict but doesn't strongly support character specificity.",
                "scale_2": "Clear vocal mismatch with established character traits (wrong age range, energy level).",
                "scale_1": "Fundamental identity confusion (wrong gender, completely inappropriate vocal characteristics)."
            },
            {
                "name": "VoiceIdentityMatching",
                "description": "How convincingly the voice represents the intended speaker.",
                "scale_5": "Unmistakably the intended character; vocal DNA perfectly matches established identity.",
                "scale_4": "Believable representation with small inconsistencies in vocal age or character energy.",
                "scale_3": "Could plausibly be various characters; lacks distinctive vocal fingerprint.",
                "scale_2": "Significant vocal disconnect from character expectations (inappropriate maturity, timbre).",
                "scale_1": "Completely wrong speaker representation that breaks character immersion."
            }
        ],
        
        "trait_embodiment": [
            {
                "name": "TraitEmbodiment",
                "description": "Presence of profile traits (positive OR opposite).",
                "scale_5": "Core traits audible and believable (positive or opposite direction if meaningful).",
                "scale_4": "Most traits present; one weak or missing.",
                "scale_3": "Traits faint; delivery neutral.",
                "scale_2": "Traits clash with profile or unclear.",
                "scale_1": "No discernible traits or fully off-profile."
            },
            {
                "name": "TraitEmbodiment",
                "description": "Manifestation of character traits in vocal delivery (including opposite traits).",
                "scale_5": "Character traits shine through authentically, whether reinforcing or meaningfully contrasting established profile.",
                "scale_4": "Most personality elements evident; minor trait weak or insufficiently expressed.",
                "scale_3": "Personality traits barely perceptible; emotionally and characterologically neutral.",
                "scale_2": "Traits conflict with established profile or remain frustratingly ambiguous.",
                "scale_1": "Complete absence of recognizable character traits; generic vocal performance."
            },
            {
                "name": "TraitEmbodiment",
                "description": "How well vocal delivery captures the speaker's defining personality features.",
                "scale_5": "Perfectly captures character essence; traits feel organic whether confirming or contrasting profile expectations.",
                "scale_4": "Strong trait representation with one characteristic underplayed or missing entirely.",
                "scale_3": "Minimal personality coloring; traits present but require careful listening to detect.",
                "scale_2": "Personality traits contradict known character or exist in confusing, unclear ways.",
                "scale_1": "Zero personality specificity; could be any character speaking these lines."
            }
        ],
        
        "local_scene_fit": [
            {
                "name": "LocalSceneFit",
                "description": "Speech content and tone suit the immediate scene.",
                "scale_5": "Tone/language align perfectly with immediate objective & mood.",
                "scale_4": "Minor mismatch but scene still works.",
                "scale_3": "Neutral; neither supports nor harms the scene.",
                "scale_2": "Somewhat confusing or tone-deaf.",
                "scale_1": "Directly contradicts scene stakes."
            },
            {
                "name": "LocalSceneFit",
                "description": "How well the delivery serves the current scene's requirements.",
                "scale_5": "Flawless integration with scene objectives, mood, and immediate dramatic needs.",
                "scale_4": "Good scene support with minor tonal inconsistencies that don't damage overall effectiveness.",
                "scale_3": "Adequately neutral; doesn't enhance scene but doesn't actively work against it.",
                "scale_2": "Noticeably inappropriate for scene context; creates mild confusion or awkwardness.",
                "scale_1": "Completely undermines scene goals; actively works against established dramatic stakes."
            },
            {
                "name": "LocalSceneFit",
                "description": "Compatibility between vocal delivery and immediate dramatic context.",
                "scale_5": "Seamless scene integration; vocal choices amplify and support every aspect of the moment.",
                "scale_4": "Strong scene compatibility with small elements that don't quite mesh perfectly.",
                "scale_3": "Scene-neutral delivery; functionally adequate without adding particular value.",
                "scale_2": "Mild scene discord; delivery choices create some contextual friction.",
                "scale_1": "Scene-breaking delivery that fundamentally contradicts the established dramatic situation."
            }
        ],
        
        "global_story_fit": [
            {
                "name": "GlobalStoryFit",
                "description": "Consistency with overarching story / profile.",
                "scale_5": "Consistent with long-term character arc and established habits.",
                "scale_4": "Slightly atypical but plausible growth.",
                "scale_3": "Generic filler; little personality reference.",
                "scale_2": "Conflicts with known backstory.",
                "scale_1": "Breaks established identity entirely."
            },
            {
                "name": "GlobalStoryFit",
                "description": "Alignment with broader character development and story continuity.",
                "scale_5": "Perfect harmony with character's established arc, personality evolution, and behavioral patterns.",
                "scale_4": "Mostly consistent with minor character growth that extends naturally from known traits.",
                "scale_3": "Bland, non-specific delivery that neither reinforces nor contradicts character history.",
                "scale_2": "Notable inconsistency with established character background or previous behavior patterns.",
                "scale_1": "Complete character betrayal; delivery fundamentally contradicts established identity and story logic."
            },
            {
                "name": "GlobalStoryFit",
                "description": "How well the delivery maintains consistency with the speaker's established identity.",
                "scale_5": "Flawless character coherence; builds naturally on established personality and story trajectory.",
                "scale_4": "Generally consistent character representation with believable but slight personality expansion.",
                "scale_3": "Generic character expression; safe but adds little to established personality understanding.",
                "scale_2": "Character inconsistency that raises questions about identity or story continuity.",
                "scale_1": "Completely breaks character; delivery could not plausibly come from the established speaker."
            }
        ]
    }
    
    return variations

def generate_system_prompt_variations() -> List[str]:
    """Generate variations of the system prompt."""
    return [
        "You are an expert evaluator of speech delivery and storytelling.",
        "You are a professional assessor specializing in speech performance and narrative evaluation.",
        "You are an experienced critic of vocal delivery and storytelling techniques.",
        "You are a skilled analyzer of speech quality and narrative performance.",
        "You are a specialist in evaluating vocal expression and storytelling effectiveness.",
        "You are an authority on speech assessment and narrative delivery evaluation.",
        "You are a professional reviewer of vocal performance and storytelling craft.",
        "You are an expert judge of speech delivery quality and narrative presentation.",
        "You are a seasoned evaluator of vocal expression and storytelling performance.",
        "You are a professional assessor of speech delivery and narrative storytelling."
    ]

def generate_output_schema_variations() -> List[str]:
    """Generate variations of the output schema instruction."""
    return [
        "OUTPUT SCHEMA \n<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>",
        "RESPONSE FORMAT \n<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>",
        "OUTPUT STRUCTURE \n<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>",
        "EXPECTED OUTPUT \n<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>",
        "RETURN FORMAT \n<answer>[PitchDynamics]1-5[RhythmicNaturalness]1-5[StressEmphasis]1-5[EmotionAccuracy]1-5[EmotionIntensity]1-5[EmotionalDynamicRange]1-5[VoiceIdentityMatching]1-5[TraitEmbodiment]1-5[LocalSceneFit]1-5[GlobalStoryFit]1-5</answer>"
    ]

def create_prompt_variation(
    system_prompt: str,
    base_header: str,
    dimension_set: Dict[str, Dict[str, str]],
    output_schema: str,
    style: str = "detailed"
) -> Dict[str, str]:
    """Create a single prompt variation."""
    
    if style == "detailed":
        # Create detailed version similar to general_prompt_detailed
        detailed_content = f"""**WHAT TO DO**  
1. **Read first:** Global Profile ➜ Local Scene.  
2. **Play** the audio **while skimming** the transcript (transcript may be imperfect).  
3. **Focus only on the target speaker.**  
   • Ignore background SFX / music / other voices except where they influence the target character's timing or tone.  
4. **Apply the rubric below EXACTLY** and return **only** as the example – no prose, line-breaks, or extra keys.  
5. **Scale is 1 (poor) … 5 (excellent).** Use integers only.  

**DIMENSIONS & RULES**  
For every dimension, 5 = best possible, 1 = worst. Use these anchor descriptions to choose the closest integer:

"""
        # Add dimension details
        for i, (dim_key, dim_data) in enumerate(dimension_set.items(), 1):
            detailed_content += f"### {i}. {dim_data['name']}  \n"
            detailed_content += f"|5|{dim_data['scale_5']}|\n"
            detailed_content += f"|4|{dim_data['scale_4']}|\n"
            detailed_content += f"|3|{dim_data['scale_3']}|\n"
            detailed_content += f"|2|{dim_data['scale_2']}|\n"
            detailed_content += f"|1|{dim_data['scale_1']}|\n\n"
        
        detailed_content += """**Penalty:** If the speaker is clearly the wrong person, set VoiceIdentityMatching = 1 and TraitEmbodiment = 1 (even if other scores are higher).
**Present results in a fixed order:** PitchDynamics, RhythmicNaturalness, StressEmphasis, EmotionAccuracy, EmotionIntensity, EmotionalDynamicRange, VoiceIdentityMatching, TraitEmbodiment, LocalSceneFit, GlobalStoryFit.

"""
        detailed_content += output_schema
        
        return {
            "system_prompt": system_prompt,
            "general_prompt_detailed": detailed_content,
            "dimension_variations": dimension_set
        }
    
    else:
        # Create simple version similar to general_prompt
        simple_content = f"""Follow the rubric **exactly** and return **only** as the example —no extra text.

RUBRIC (all 10 dimensions, 1 = poor … 5 = excellent)
"""
        # Add dimension details
        for dim_key, dim_data in dimension_set.items():
            simple_content += f"• {dim_data['name']}  {dim_data['description']}  \n"
        
        simple_content += """• TraitEmbodiment  Presence (positive **or opposite**) of profile traits.  
  – Keep a trait if audible either way; opposite ⇒ low score.  
  – If none remain, set `"traits":"N/A"` and TraitEmbodiment = 1.  

Penalty Explicit speaker mismatch ⇒ VoiceIdentityMatching = 1 and TraitEmbodiment = 1.
Present results in a fixed order: PitchDynamics, RhythmicNaturalness, StressEmphasis, EmotionAccuracy, EmotionIntensity, EmotionalDynamicRange, VoiceIdentityMatching, TraitEmbodiment, LocalSceneFit, GlobalStoryFit.

"""
        simple_content += output_schema
        
        return {
            "system_prompt": system_prompt,
            "general_prompt": simple_content,
            "dimension_variations": dimension_set
        }

def generate_all_variations() -> Dict[str, List[Dict]]:
    """Generate 100 variations of all prompt types."""
    
    base_headers = generate_base_header_variations()
    system_prompts = generate_system_prompt_variations()
    dimension_variations = generate_dimension_variations()
    output_schemas = generate_output_schema_variations()
    
    all_variations = {
        "detailed_variations": [],
        "simple_variations": [],
        "dimension_prompts": [],
        "caption_variations": [],
        "emotion_variations": []
    }
    
    # Generate detailed variations (25 variations)
    for i in range(100):
        system_prompt = random.choice(system_prompts)
        output_schema = random.choice(output_schemas)
        
        # Create dimension set by selecting one variation for each dimension
        dimension_set = {}
        for dim_key, variations in dimension_variations.items():
            dimension_set[dim_key] = random.choice(variations)
        
        variation = create_prompt_variation(
            system_prompt, "", dimension_set, output_schema, "detailed"
        )
        variation["id"] = f"detailed_{i+1}"
        all_variations["detailed_variations"].append(variation)
    
    # Generate simple variations (25 variations)
    for i in range(100):
        system_prompt = random.choice(system_prompts)
        output_schema = random.choice(output_schemas)
        
        dimension_set = {}
        for dim_key, variations in dimension_variations.items():
            dimension_set[dim_key] = random.choice(variations)
        
        variation = create_prompt_variation(
            system_prompt, "", dimension_set, output_schema, "simple"
        )
        variation["id"] = f"simple_{i+1}"
        all_variations["simple_variations"].append(variation)
    
    # Generate individual dimension prompt variations (30 variations)
    for dim_key in dimension_variations.keys():
        for base_header in base_headers:
            for dim_variation in dimension_variations[dim_key]:
        
                prompt_text = base_header + f"Dimension: {dim_variation['name']} — {dim_variation['description']}\n\n"
                prompt_text += f"5: {dim_variation['scale_5']}\n"
                prompt_text += f"4: {dim_variation['scale_4']}\n"
                prompt_text += f"3: {dim_variation['scale_3']}\n"
                prompt_text += f"2: {dim_variation['scale_2']}\n"
                prompt_text += f"1: {dim_variation['scale_1']}\n"
                prompt_text += f"Example output: <answer>{random.randint(1, 5)}</answer>"
        
                variation = {
                    "id": f"dimension_{i+1}",
                    "dimension_name": dim_key,
                    "prompt_text": prompt_text
                }
                all_variations["dimension_prompts"].append(variation)
   
    ''' 
    # Generate caption variations (10 variations)
    for i in range(10):
        base_variation = random.choice(all_variations["simple_variations"])
        caption_variation = base_variation.copy()
        caption_variation["id"] = f"caption_{i+1}"
        caption_variation["general_prompt"] += "\n\nIn addition to the above rubric, please also provide the audio caption in <think> </think> and the emotion caption in <think> </think> before the answer."
        all_variations["caption_variations"].append(caption_variation)
    
    # Generate emotion variations (10 variations)
    for i in range(10):
        base_variation = random.choice(all_variations["simple_variations"])
        emotion_variation = base_variation.copy()
        emotion_variation["id"] = f"emotion_{i+1}"
        emotion_variation["general_prompt"] += "\n\nIn addition to the above rubric, please also provide the emotion caption in <think> </think> before the answer."
        all_variations["emotion_variations"].append(emotion_variation)'''
    
    return all_variations

def main():
    """Generate and save prompt variations to JSON file."""
    print("Generating 100 prompt variations...")
    
    variations = generate_all_variations()
    
    # Add metadata
    output_data = {
        "metadata": {
            "total_variations": sum(len(v) for v in variations.values()),
            "generation_date": "2025-07-23",
            "description": "Generated variations of speech evaluation prompts"
        },
        "variations": variations
    }
    
    # Save to JSON file
    with open("prompt_variations.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {output_data['metadata']['total_variations']} variations:")
    for category, items in variations.items():
        print(f"  {category}: {len(items)} variations")
    
    print("\nSaved to prompt_variations.json")

if __name__ == "__main__":
    main()
