from deep_translator import GoogleTranslator

def translate_to_english(text):
    dictionary = {
        "madhuram": "sweet",
        "sneham": "love",
        "pranayam": "romance",
        "hridayam": "heart",
        "swapnam": "dream",
        "raagam": "melody",
        "jeevan": "life",
        "nila": "moon"
    }

    return dictionary.get(text.lower(), text)
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text