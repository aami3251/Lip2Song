from gtts import gTTS
from pydub import AudioSegment
import os

def generate_audio(text):
    try:
        # -------------------------
        # 1. Generate Voice
        # -------------------------
        tts = gTTS(text=text, lang='en')
        tts.save("voice.mp3")

        voice = AudioSegment.from_file("voice.mp3")

        # -------------------------
        # 2. Slight Singing Feel
        # -------------------------
        voice = voice._spawn(voice.raw_data, overrides={
            "frame_rate": int(voice.frame_rate * 0.92)
        }).set_frame_rate(voice.frame_rate)

        # -------------------------
        # 3. Load Background Music
        # -------------------------
        bg = AudioSegment.from_file("background.mp3")

        # Make it same length
        while len(bg) < len(voice):
            bg += bg

        bg = bg[:len(voice)]

        # Make background soft
        bg = bg - 18

        # -------------------------
        # 4. Combine (THIS IS THE MAIN PART)
        # -------------------------
        final_audio = bg.overlay(voice)

        # -------------------------
        # 5. Smooth Ending
        # -------------------------
        final_audio = final_audio.fade_in(300).fade_out(700)

        # -------------------------
        # 6. Export
        # -------------------------
        final_audio.export("english_output.wav", format="wav")

        # Cleanup
        if os.path.exists("voice.mp3"):
            os.remove("voice.mp3")

        return "english_output.wav"

    except Exception as e:
        print("Error:", e)
        return None