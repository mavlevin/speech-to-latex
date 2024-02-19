"""

Notice: research code; mediocre quality and to be used on trusted input only.

"""
from pathlib import Path
from openai import OpenAI
import re
import os

def text_to_speech(client, text, speech_file_path):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="onyx",
        input=text
    )
    response.stream_to_file(speech_file_path)

def is_text_tag(tag):
    return re.match(r'^.*_text_\d+$', tag) is not None

def get_equation_text_options(tagged_equation_texts_file_path):
    eq_texts = {}
    with open(tagged_equation_texts_file_path, "r") as eq_file:
        for line in eq_file:
            if len(line) > len("Equation_num_"):
                # line has tag
                tag = line.split(": \"")[0]
                value = line[len(tag)+3:] # +5 to get rid of sepeartor
                if is_text_tag(tag):
                    eq_texts[tag] = value
    
    return eq_texts

def tts_dict(client, input_dict, root_save_path, allow_cached=True):

    for tag, text in input_dict.items():
        save_path = Path(__file__).parent / root_save_path / f"{tag}.mp3"
        if allow_cached and os.path.exists(save_path):
            print(f"found cached {tag}")
            continue
            
        text_to_speech(client, text, save_path)
        print(f"tts'ed {tag}")

def main():
    print("Starting.")

    eq_texts = get_equation_text_options("./tagged_plain_english_dataset.txt")

    print(f"Loaded {len(eq_texts)} equation texts.")

    client = OpenAI()

    tts_dict(client, eq_texts, "audiodb")

    print("Done.")

if __name__ == "__main__":
    main()