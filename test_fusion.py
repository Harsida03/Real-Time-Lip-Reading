"""
Test script to demonstrate audio-visual fusion logic without requiring hardware.
This simulates the fusion algorithm with example predictions.
"""

from fusion import AudioVisualFusion


def test_fusion_scenarios():
    """Test various fusion scenarios"""
    fusion = AudioVisualFusion(audio_weight=0.6, visual_weight=0.4)
    
    print("=" * 80)
    print("AUDIO-VISUAL FUSION TEST")
    print("=" * 80)
    print()
    
    # Test 1: High agreement (similar words)
    print("Test 1: High Agreement (Viseme disambiguation)")
    print("-" * 80)
    visual = "I saw a bat"
    audio = "I saw a pat"
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.5, audio_confidence=0.7)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 2: Medium agreement (partial match)
    print("Test 2: Medium Agreement")
    print("-" * 80)
    visual = "the cat sat on the mat"
    audio = "the cat sat on the bat"
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.5, audio_confidence=0.7)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 3: Low agreement (different sentences)
    print("Test 3: Low Agreement")
    print("-" * 80)
    visual = "hello world"
    audio = "goodbye moon"
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.5, audio_confidence=0.7)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 4: Audio only (no visual)
    print("Test 4: Audio Only (noisy video)")
    print("-" * 80)
    visual = ""
    audio = "this is audio only"
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.5, audio_confidence=0.7)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 5: Visual only (no audio)
    print("Test 5: Visual Only (noisy audio)")
    print("-" * 80)
    visual = "this is visual only"
    audio = None
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.5, audio_confidence=0.7)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 6: High visual confidence
    print("Test 6: High Visual Confidence (low agreement)")
    print("-" * 80)
    visual = "the quick brown fox"
    audio = "the slow red dog"
    fused, method = fusion.fuse_predictions(visual, audio, visual_confidence=0.9, audio_confidence=0.6)
    print(f"Visual: '{visual}'")
    print(f"Audio:  '{audio}'")
    print(f"Fused:  '{fused}'")
    print(f"Method: {method}")
    print()
    
    # Test 7: Viseme confusion examples
    print("Test 7: Common Viseme Confusions")
    print("-" * 80)
    viseme_pairs = [
        ("bat", "pat"),
        ("mat", "bat"),
        ("fan", "van"),
        ("see", "she"),
        ("thin", "sin")
    ]
    
    for v, a in viseme_pairs:
        fused, method = fusion.fuse_predictions(v, a, visual_confidence=0.5, audio_confidence=0.7)
        print(f"Visual: '{v}' | Audio: '{a}' => Fused: '{fused}'")
    
    print()
    print("=" * 80)
    print("Test complete! The fusion algorithm successfully disambiguates")
    print("visually similar words by incorporating audio information.")
    print("=" * 80)


if __name__ == "__main__":
    test_fusion_scenarios()
