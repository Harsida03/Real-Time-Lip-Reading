"""
Audio-Visual Fusion Module
Combines audio and visual predictions using weighted confidence scores.
This module has no hardware dependencies and can be used for testing.
"""

from typing import Tuple, Optional


class AudioVisualFusion:
    """
    Combines audio and visual speech recognition predictions using weighted scores.
    """
    
    def __init__(self, audio_weight=0.6, visual_weight=0.4):
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        assert abs((audio_weight + visual_weight) - 1.0) < 0.01, "Weights must sum to 1.0"
        
    def fuse_predictions(
        self, 
        visual_text: str, 
        audio_text: Optional[str],
        visual_confidence: float = 0.5,
        audio_confidence: float = 0.7
    ) -> Tuple[str, str]:
        """
        Fuse audio and visual predictions based on similarity and weighted confidence.
        
        Args:
            visual_text: Text from visual speech recognition (lip-reading)
            audio_text: Text from audio speech recognition (may be None)
            visual_confidence: Confidence score for visual prediction (0.0 to 1.0)
            audio_confidence: Confidence score for audio prediction (0.0 to 1.0)
            
        Returns:
            Tuple of (fused_text, fusion_method_description)
        """
        if not audio_text:
            return visual_text, "visual_only (no audio)"
        
        if not visual_text or len(visual_text.strip()) == 0:
            return audio_text, "audio_only (no visual)"
        
        visual_words = visual_text.lower().split()
        audio_words = audio_text.lower().split()
        
        if len(audio_words) == 0:
            return visual_text, "visual_only (empty audio)"
        
        if len(visual_words) == 0:
            return audio_text, "audio_only (empty visual)"
        
        similarity = self._calculate_similarity(visual_words, audio_words)
        
        weighted_audio_score = audio_confidence * self.audio_weight
        weighted_visual_score = visual_confidence * self.visual_weight
        
        if similarity > 0.8:
            if weighted_audio_score > weighted_visual_score:
                return audio_text, f"high_agreement_audio (sim: {similarity:.2f}, scores: A={weighted_audio_score:.2f} V={weighted_visual_score:.2f})"
            else:
                return visual_text, f"high_agreement_visual (sim: {similarity:.2f}, scores: A={weighted_audio_score:.2f} V={weighted_visual_score:.2f})"
        elif similarity > 0.3:
            fused = self._word_level_fusion(visual_words, audio_words, weighted_audio_score, weighted_visual_score)
            return fused, f"word_level_fusion (sim: {similarity:.2f}, scores: A={weighted_audio_score:.2f} V={weighted_visual_score:.2f})"
        else:
            if weighted_visual_score > weighted_audio_score:
                return visual_text, f"low_agreement_visual (sim: {similarity:.2f}, scores: A={weighted_audio_score:.2f} V={weighted_visual_score:.2f})"
            else:
                return audio_text, f"low_agreement_audio (sim: {similarity:.2f}, scores: A={weighted_audio_score:.2f} V={weighted_visual_score:.2f})"
    
    def _calculate_similarity(self, words1: list, words2: list) -> float:
        """Calculate Jaccard similarity between two word lists."""
        if not words1 or not words2:
            return 0.0
        
        set1 = set(words1)
        set2 = set(words2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _word_level_fusion(self, visual_words: list, audio_words: list, audio_score: float, visual_score: float) -> str:
        """
        Fuse predictions at word level using weighted scores.
        When words disagree, chooses based on modality scores.
        """
        max_len = max(len(visual_words), len(audio_words))
        fused_words = []
        
        for i in range(max_len):
            if i < len(audio_words) and i < len(visual_words):
                if audio_words[i] == visual_words[i]:
                    fused_words.append(audio_words[i])
                else:
                    if audio_score > visual_score:
                        fused_words.append(audio_words[i])
                    else:
                        fused_words.append(visual_words[i])
            elif i < len(audio_words):
                fused_words.append(audio_words[i])
            elif i < len(visual_words):
                fused_words.append(visual_words[i])
        
        return ' '.join(fused_words)
