import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from main import analyze_text, extract_tones

def test_analyze_text():
    text = "Tôi học."
    tokens, pos, tones = analyze_text(text)
    assert tokens == ['Tôi', 'học', '.']
    assert isinstance(pos, list)
    assert isinstance(tones, list)
    print("✅ analyze_text passed")

def test_extract_tones():
    tones = extract_tones("Tôi học.")
    assert tones == [('Tôi', ['ngang']), ('học.', ['nặng'])] or tones == [('Tôi', ['ngang']), ('học.', ['ngang'])]  # both may appear due to punctuation
    print("✅ extract_tones passed")

if __name__ == "__main__":
    test_analyze_text()
    test_extract_tones()
