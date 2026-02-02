"""
REAL-TIME HUMAN-FRIENDLY EXPLAINABILITY FOR NEW USER POSTS
===========================================================
Generates descriptive, human-readable explanations for individual posts
in real-time, similar to explainability_descriptive_user_v2.py but for
new/unseen posts coming from the inference pipeline.
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional


class RealtimeExplainer:
    """
    Generates human-friendly explanations for new posts in real-time.
    
    This class takes the raw detection results from the inference pipeline
    and transforms them into descriptive, easy-to-understand explanations
    similar to the batch explainability layer.
    """
    
    def __init__(self):
        """Initialize thresholds and mappings for explanation generation."""
        
        # Thresholds (calibrated from batch analysis)
        self.thresholds = {
            'contradiction': {
                'critical': 0.90,
                'high': 0.75,
                'moderate': 0.50,
                'low': 0.25
            },
            'emotion_intensity': {
                'strong': 0.75,
                'moderate': 0.50,
                'calm': 0.25
            },
            'suspicion': {
                'critical': 0.85,
                'high': 0.70,
                'moderate': 0.55,
                'low': 0.40
            },
            'confidence': {
                'very_high': 0.80,
                'high': 0.60,
                'moderate': 0.40,
                'low': 0.20
            }
        }
        
        # Emoji mappings for visual clarity
        self.emojis = {
            'critical': '🚨',
            'high': '⚠️',
            'moderate': '🟡',
            'low': '✅',
            'emotion_strong': '💥',
            'emotion_calm': '😐',
            'emotion_moderate': '🙂',
            'narrative': '🔁',
            'new_content': '🆕',
            'temporal_rapid': '⏱️',
            'temporal_slow': '🐌',
            'modality_text': '📝',
            'modality_image': '🖼️',
            'modality_meta': '📊'
        }
    
    def generate_explanation(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete human-friendly explanation from detection results.
        
        Args:
            detection_results: Output from LightweightFakeNewsDetector.process_user_post()
            
        Returns:
            Dictionary with structured explanation components
        """
        # Extract key components
        post_id = detection_results.get('post_id', 'unknown')
        username = detection_results.get('username', 'unknown')
        timestamp = detection_results.get('timestamp', datetime.now().isoformat())
        final_verdict = detection_results.get('final_verdict', {})
        
        # Build explanation sections
        sections = {
            'header': self._generate_header(post_id, username, timestamp, final_verdict),
            'overall_verdict': self._explain_overall_verdict(final_verdict),
            'contradiction_analysis': self._explain_contradiction(detection_results),
            'emotional_analysis': self._explain_emotions(detection_results),
            'modality_breakdown': self._explain_modalities(detection_results),
            'content_patterns': self._explain_content_patterns(detection_results),
            'risk_indicators': self._identify_risk_indicators(detection_results),
            'recommendation': self._generate_recommendation(final_verdict)
        }
        
        # Combine into full narrative
        full_narrative = self._build_full_narrative(sections)
        
        return {
            'post_id': post_id,
            'username': username,
            'timestamp': timestamp,
            'sections': sections,
            'full_narrative': full_narrative,
            'structured_data': self._extract_structured_data(detection_results)
        }
    
    def _generate_header(self, post_id: str, username: str, timestamp: str, 
                        verdict: Dict) -> str:
        """Generate header section."""
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = timestamp
        
        score = verdict.get('score', 0.5)
        label = verdict.get('label', 'unknown').upper()
        confidence = verdict.get('confidence', 0.0)
        
        # Select emoji based on verdict
        if label == 'FAKE':
            verdict_emoji = self.emojis['critical'] if score > 0.7 else self.emojis['high']
        else:
            verdict_emoji = self.emojis['low']
        
        return (
            f"{verdict_emoji} Post {post_id} by {username} at {time_str}\n"
            f"VERDICT: {label} (Score: {score:.2f}, Confidence: {confidence*100:.1f}%)"
        )
    
    def _explain_overall_verdict(self, verdict: Dict) -> str:
        """Explain the overall verdict."""
        score = verdict.get('score', 0.5)
        label = verdict.get('label', 'unknown')
        confidence = verdict.get('confidence', 0.0)
        
        if label == 'fake':
            if score > 0.85:
                severity = "CRITICAL"
                emoji = self.emojis['critical']
                desc = "This post exhibits multiple strong indicators of fake news or manipulation."
            elif score > 0.70:
                severity = "HIGH"
                emoji = self.emojis['high']
                desc = "This post shows significant characteristics of deceptive content."
            elif score > 0.55:
                severity = "MODERATE"
                emoji = self.emojis['moderate']
                desc = "This post displays some concerning patterns that warrant further verification."
            else:
                severity = "LOW"
                emoji = self.emojis['low']
                desc = "This post shows minor suspicious patterns but is likely genuine."
        else:
            severity = "AUTHENTIC"
            emoji = self.emojis['low']
            desc = "This post appears to be genuine with no significant deception indicators."
        
        confidence_level = (
            "very high" if confidence > 0.8 else
            "high" if confidence > 0.6 else
            "moderate" if confidence > 0.4 else
            "low"
        )
        
        return (
            f"{emoji} Overall Assessment: {severity}\n"
            f"Deception Score: {score:.2f}/1.00 ({severity} risk)\n"
            f"Model Confidence: {confidence*100:.1f}% ({confidence_level})\n\n"
            f"Interpretation: {desc}"
        )
    
    def _explain_contradiction(self, results: Dict) -> str:
        """Explain contradiction/mismatch analysis."""
        explanation_parts = []
        
        # Emotional contradiction from emotion-aware model
        if 'emotion_aware' in results:
            ea = results['emotion_aware']
            congruence = ea.get('emotional_congruence', 0)
            mismatch_mag = ea.get('mismatch_magnitude', 0)
            
            # Normalize mismatch (typical range 0-2, normalize to 0-1)
            norm_mismatch = min(mismatch_mag / 2.0, 1.0)
            
            if norm_mismatch > self.thresholds['contradiction']['critical']:
                emoji = self.emojis['critical']
                level = "CRITICAL"
                desc = "Strong conflict between this post and other content — high deception risk."
            elif norm_mismatch > self.thresholds['contradiction']['high']:
                emoji = self.emojis['high']
                level = "HIGH"
                desc = "This post contradicts other content, which may confuse readers."
            elif norm_mismatch > self.thresholds['contradiction']['moderate']:
                emoji = self.emojis['moderate']
                level = "MODERATE"
                desc = "Some inconsistency with surrounding content patterns."
            else:
                emoji = self.emojis['low']
                level = "LOW"
                desc = "This post aligns reasonably well with expected patterns."
            
            explanation_parts.append(
                f"{emoji} Contradiction Score: {norm_mismatch:.2f} ({level})\n"
                f"   {desc}\n"
                f"   Emotional Congruence: {congruence:.2f} "
                f"({'positive alignment' if congruence > 0 else 'negative alignment'})"
            )
        
        return "\n".join(explanation_parts) if explanation_parts else "✅ No contradiction analysis available"
    
    def _explain_emotions(self, results: Dict) -> str:
        """Explain emotional analysis."""
        explanation_parts = []
        
        # VAD analysis
        if 'explanation' in results and 'vad_analysis' in results['explanation']:
            vad = results['explanation']['vad_analysis']
            valence = vad.get('text_valence', 0.5)
            arousal = vad.get('text_arousal', 0.5)
            dominance = vad.get('text_dominance', 0.5)
            
            # Compute overall intensity
            intensity = np.sqrt(valence**2 + arousal**2 + dominance**2) / np.sqrt(3)
            
            if intensity > self.thresholds['emotion_intensity']['strong']:
                emoji = self.emojis['emotion_strong']
                level = "STRONG"
                desc = "Likely intended to grab attention or provoke reactions."
            elif intensity < self.thresholds['emotion_intensity']['calm']:
                emoji = self.emojis['emotion_calm']
                level = "CALM"
                desc = "Tone is neutral and measured."
            else:
                emoji = self.emojis['emotion_moderate']
                level = "MODERATE"
                desc = "Balanced emotional tone."
            
            explanation_parts.append(
                f"{emoji} Emotional Intensity: {intensity:.2f} ({level})\n"
                f"   {desc}\n"
                f"   • Valence (positive/negative): {valence:.2f} "
                f"({'positive' if valence > 0.5 else 'negative'})\n"
                f"   • Arousal (calm/excited): {arousal:.2f} "
                f"({'high arousal' if arousal > 0.5 else 'calm'})\n"
                f"   • Dominance (control/submissive): {dominance:.2f} "
                f"({'dominant' if dominance > 0.5 else 'submissive'})"
            )
        
        # Mixed affect from emotion-aware model
        if 'emotion_aware' in results:
            mixed_affect = results['emotion_aware'].get('mixed_affect_score', 0)
            if mixed_affect > 0.5:
                explanation_parts.append(
                    f"\n⚠️ Mixed Emotional Signals: {mixed_affect:.2f}\n"
                    f"   Post shows conflicting emotional cues, which may indicate manipulation."
                )
        
        return "\n".join(explanation_parts) if explanation_parts else "✅ No emotional analysis available"
    
    def _explain_modalities(self, results: Dict) -> str:
        """Explain which modalities contributed most to detection."""
        explanation_parts = []
        
        # Emotion-aware weights
        if 'emotion_aware' in results and 'emotion_weights' in results['emotion_aware']:
            weights = results['emotion_aware']['emotion_weights']
            text_w = weights.get('text', 0)
            image_w = weights.get('image', 0)
            meta_w = weights.get('meta', 0)
            
            # Find dominant modality
            modalities = [
                ('text', text_w, self.emojis['modality_text'], 'Text'),
                ('image', image_w, self.emojis['modality_image'], 'Image'),
                ('meta', meta_w, self.emojis['modality_meta'], 'Metadata')
            ]
            modalities.sort(key=lambda x: x[1], reverse=True)
            
            dominant_name, dominant_weight, dominant_emoji, dominant_label = modalities[0]
            
            explanation_parts.append(
                f"{dominant_emoji} Detection primarily driven by {dominant_label} signals ({dominant_weight*100:.1f}% contribution)\n"
                f"\nModality Breakdown:"
            )
            
            for name, weight, emoji, label in modalities:
                bar_length = int(weight * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                explanation_parts.append(
                    f"   {emoji} {label:8s} [{bar}] {weight*100:5.1f}%"
                )
        
        # Adaptive fusion weights (if available)
        if 'adaptive_fusion' in results and 'modality_weights' in results['adaptive_fusion']:
            weights = results['adaptive_fusion']['modality_weights']
            explanation_parts.append(
                f"\nAlternate Model Weighting:\n"
                f"   📝 Text: {weights.get('text', 0)*100:.1f}%\n"
                f"   🖼️ Image: {weights.get('image', 0)*100:.1f}%\n"
                f"   📊 Meta: {weights.get('meta', 0)*100:.1f}%"
            )
        
        return "\n".join(explanation_parts) if explanation_parts else "✅ No modality analysis available"
    
    def _explain_content_patterns(self, results: Dict) -> str:
        """Explain content-level patterns detected."""
        explanation_parts = []
        metadata = results.get('metadata', {})
        
        # URLs
        url_count = metadata.get('urls_count', 0)
        if url_count > 5:
            explanation_parts.append(
                f"⚠️ High URL count ({url_count}) detected — may indicate spam or link farming"
            )
        elif url_count > 2:
            explanation_parts.append(
                f"🟡 Moderate URL usage ({url_count}) — verify link legitimacy"
            )
        elif url_count > 0:
            explanation_parts.append(
                f"✅ Minimal URLs ({url_count}) present"
            )
        
        # Hashtags
        hashtag_count = metadata.get('hashtags_count', 0)
        if hashtag_count > 10:
            explanation_parts.append(
                f"⚠️ Excessive hashtags ({hashtag_count}) — potential visibility manipulation"
            )
        elif hashtag_count > 5:
            explanation_parts.append(
                f"🟡 High hashtag usage ({hashtag_count})"
            )
        
        # Mentions
        mention_count = metadata.get('mentions_count', 0)
        if mention_count > 10:
            explanation_parts.append(
                f"⚠️ Many user mentions ({mention_count}) — check for targeted harassment or spam"
            )
        
        # Emojis
        emoji_count = metadata.get('emojis_count', 0)
        if emoji_count > 10:
            explanation_parts.append(
                f"🟡 Heavy emoji usage ({emoji_count}) — may indicate emotional manipulation"
            )
        elif emoji_count > 5:
            explanation_parts.append(
                f"Moderate emoji usage ({emoji_count})"
            )
        
        # Objects detected in image
        obj_count = metadata.get('objects_detected', 0)
        if obj_count > 0:
            explanation_parts.append(
                f"🖼️ Image contains {obj_count} detected objects"
            )
        
        return "\n".join(explanation_parts) if explanation_parts else "✅ No unusual content patterns detected"
    
    def _identify_risk_indicators(self, results: Dict) -> str:
        """Identify specific risk indicators."""
        indicators = []
        
        # High suspicion score
        if 'emotion_aware' in results:
            pred = results['emotion_aware'].get('prediction', 0)
            if pred > 0.85:
                indicators.append("🚨 CRITICAL: Very high deception score")
            elif pred > 0.70:
                indicators.append("⚠️ HIGH: Elevated deception indicators")
        
        # Emotional manipulation
        if 'emotion_aware' in results:
            ea = results['emotion_aware']
            if ea.get('mismatch_magnitude', 0) > 1.0:
                indicators.append("⚠️ Emotional contradiction detected")
            if ea.get('mixed_affect_score', 0) > 0.6:
                indicators.append("⚠️ Mixed emotional signals (possible manipulation)")
        
        # Content spam indicators
        metadata = results.get('metadata', {})
        if metadata.get('urls_count', 0) > 5:
            indicators.append("⚠️ High URL density (link farming)")
        if metadata.get('hashtags_count', 0) > 10:
            indicators.append("⚠️ Hashtag spam detected")
        if metadata.get('mentions_count', 0) > 10:
            indicators.append("⚠️ Mass mention pattern")
        
        # Low confidence warning
        if 'final_verdict' in results:
            if results['final_verdict'].get('confidence', 1.0) < 0.3:
                indicators.append("🟡 Low model confidence — manual review recommended")
        
        if not indicators:
            return "✅ No critical risk indicators detected"
        
        return "Risk Indicators:\n" + "\n".join(f"   {ind}" for ind in indicators)
    
    def _generate_recommendation(self, verdict: Dict) -> str:
        """Generate actionable recommendation."""
        score = verdict.get('score', 0.5)
        label = verdict.get('label', 'unknown')
        confidence = verdict.get('confidence', 0.0)
        
        if label == 'fake':
            if score > 0.85:
                return (
                    "🚨 RECOMMENDATION: DO NOT SHARE\n"
                    "This content shows strong indicators of fake news or manipulation. "
                    "Consider reporting to platform moderators."
                )
            elif score > 0.70:
                return (
                    "⚠️ RECOMMENDATION: VERIFY BEFORE SHARING\n"
                    "This content shows significant deception indicators. "
                    "Cross-reference with trusted sources before engagement."
                )
            elif score > 0.55:
                return (
                    "🟡 RECOMMENDATION: APPROACH WITH CAUTION\n"
                    "This content displays some suspicious patterns. "
                    "Apply critical thinking and fact-check claims."
                )
            else:
                return (
                    "✅ RECOMMENDATION: LIKELY SAFE\n"
                    "Minor concerns detected but content appears largely authentic."
                )
        else:
            if confidence > 0.7:
                return (
                    "✅ RECOMMENDATION: APPEARS AUTHENTIC\n"
                    "Content shows no significant deception indicators."
                )
            else:
                return (
                    "🟡 RECOMMENDATION: APPEARS AUTHENTIC (LOW CONFIDENCE)\n"
                    "No major red flags, but model confidence is moderate. "
                    "Standard fact-checking still recommended."
                )
    
    def _build_full_narrative(self, sections: Dict) -> str:
        """Combine all sections into a cohesive narrative."""
        narrative_parts = [
            sections['header'],
            "\n" + "="*80,
            "\n📋 OVERALL ASSESSMENT",
            "="*80,
            sections['overall_verdict'],
            "\n" + "="*80,
            "\n🔍 DETAILED ANALYSIS",
            "="*80,
            "\n💭 Contradiction Analysis:",
            "-"*80,
            sections['contradiction_analysis'],
            "\n\n🎭 Emotional Analysis:",
            "-"*80,
            sections['emotional_analysis'],
            "\n\n📊 Modality Contribution:",
            "-"*80,
            sections['modality_breakdown'],
            "\n\n📝 Content Patterns:",
            "-"*80,
            sections['content_patterns'],
            "\n" + "="*80,
            "\n⚠️ RISK ASSESSMENT",
            "="*80,
            sections['risk_indicators'],
            "\n" + "="*80,
            "\n🎯 RECOMMENDATION",
            "="*80,
            sections['recommendation'],
            "\n" + "="*80
        ]
        
        return "\n".join(narrative_parts)
    
    def _extract_structured_data(self, results: Dict) -> Dict:
        """Extract structured data for API/JSON responses."""
        verdict = results.get('final_verdict', {})
        
        structured = {
            'verdict': {
                'label': verdict.get('label', 'unknown'),
                'score': float(verdict.get('score', 0.5)),
                'confidence': float(verdict.get('confidence', 0.0)),
                'risk_level': self._compute_risk_level(verdict.get('score', 0.5))
            },
            'metadata': results.get('metadata', {}),
            'analysis': {}
        }
        
        # Add emotion-aware results
        if 'emotion_aware' in results:
            ea = results['emotion_aware']
            structured['analysis']['emotional'] = {
                'prediction': float(ea.get('prediction', 0)),
                'congruence': float(ea.get('emotional_congruence', 0)),
                'mismatch_magnitude': float(ea.get('mismatch_magnitude', 0)),
                'mixed_affect_score': float(ea.get('mixed_affect_score', 0)),
                'modality_weights': ea.get('emotion_weights', {})
            }
        
        # Add VAD
        if 'explanation' in results and 'vad_analysis' in results['explanation']:
            structured['analysis']['vad'] = results['explanation']['vad_analysis']
        
        return structured
    
    def _compute_risk_level(self, score: float) -> str:
        """Compute categorical risk level."""
        if score > 0.85:
            return "critical"
        elif score > 0.70:
            return "high"
        elif score > 0.55:
            return "moderate"
        elif score > 0.40:
            return "low"
        else:
            return "minimal"


def format_explanation_for_display(explanation: Dict) -> str:
    """
    Format explanation for clean terminal/UI display.
    
    Args:
        explanation: Output from RealtimeExplainer.generate_explanation()
        
    Returns:
        Formatted string ready for display
    """
    return explanation['full_narrative']


def format_explanation_for_json(explanation: Dict) -> Dict:
    """
    Format explanation for JSON API response.
    
    Args:
        explanation: Output from RealtimeExplainer.generate_explanation()
        
    Returns:
        Clean dictionary for JSON serialization
    """
    return {
        'post_id': explanation['post_id'],
        'username': explanation['username'],
        'timestamp': explanation['timestamp'],
        'verdict': explanation['structured_data']['verdict'],
        'metadata': explanation['structured_data']['metadata'],
        'analysis': explanation['structured_data']['analysis'],
        'summary': explanation['sections']['overall_verdict'],
        'recommendation': explanation['sections']['recommendation']
    }


# Example usage
if __name__ == "__main__":
    # Mock detection result
    mock_result = {
        'post_id': 'test_001',
        'username': 'test_user',
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'hashtags_count': 3,
            'mentions_count': 1,
            'urls_count': 2,
            'emojis_count': 5,
            'objects_detected': 2
        },
        'emotion_aware': {
            'prediction': 0.78,
            'is_fake': True,
            'confidence': 0.56,
            'emotion_weights': {'text': 0.6, 'image': 0.25, 'meta': 0.15},
            'emotional_congruence': -0.45,
            'mismatch_magnitude': 1.2,
            'mixed_affect_score': 0.62
        },
        'explanation': {
            'vad_analysis': {
                'text_valence': 0.3,
                'text_arousal': 0.8,
                'text_dominance': 0.6
            }
        },
        'final_verdict': {
            'score': 0.78,
            'label': 'fake',
            'confidence': 0.56
        }
    }
    
    explainer = RealtimeExplainer()
    explanation = explainer.generate_explanation(mock_result)
    
    print("\n" + "="*100)
    print("EXAMPLE: HUMAN-FRIENDLY EXPLANATION")
    print("="*100)
    print(format_explanation_for_display(explanation))
    
    print("\n\n" + "="*100)
    print("EXAMPLE: JSON FORMAT (for API)")
    print("="*100)
    import json
    print(json.dumps(format_explanation_for_json(explanation), indent=2))