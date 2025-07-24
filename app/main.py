from underthesea import word_tokenize, pos_tag
import unicodedata

def extract_tones(text):
    tone_marks = {
        "́": "sắc",
        "̀": "huyền",
        "̉": "hỏi",
        "̃": "ngã",
        "̣": "nặng"
    }

    def get_tone(char):
        decomposed = unicodedata.normalize('NFD', char)
        for mark in tone_marks:
            if mark in decomposed:
                return tone_marks[mark]
        return "ngang"

    tones = []
    for word in text.split():
        word_tones = set(get_tone(c) for c in word if get_tone(c) != "ngang")
        tones.append((word, list(word_tones) or ["ngang"]))
    return tones

def analyze_text(text):
    tokens = word_tokenize(text)
    pos = pos_tag(text)
    tones = extract_tones(text)
    return tokens, pos, tones

if __name__ == "__main__":
    sentence = "Tôi là người Việt Nam."
    tokens, pos, tones = analyze_text(sentence)

    print("Tokens:", tokens)
    print("POS Tags:", pos)
    print("Tones:", tones)