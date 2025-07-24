#Require package: underthesea https://github.com/undertheseanlp/underthesea/tree/main?tab=readme-ov-file

from underthesea import word_tokenize, pos_tag
import unicodedata

sentence = "Tôi là người Việt Nam"

tokens = word_tokenize(sentence)
print("Tokens: ", tokens)

pos_tags = pos_tag(sentence)
print("POS Tags: ", pos_tags)

def get_vietnamese_tone(char):
    decomposed = unicodedata.normalize('NFD', char)
    tone_marks = {'́': 'sắc', '̀': 'huyền', '̉': 'hỏi', '̃': 'ngã', '̣': 'nặng'}
    for mark in tone_marks:
        if mark in decomposed:
            return tone_marks[mark]
    return 'ngang'

def extract_tones(text):
    tones = []
    for word in text.split():
        word_tones = set()
        for char in word:
            tone = get_vietnamese_tone(char)
            if tone != 'ngang':
                word_tones.add(tone)
        tones.append((word, list(word_tones) or ['ngang']))
    return tones

# Step 5: Tone info
tones = extract_tones(sentence)
print("Tones:", tones)
