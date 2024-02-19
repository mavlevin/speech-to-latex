import whisper

from pathlib import Path
from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI()

import re

def is_text_tag(tag):
    return re.match(r'^.*_text_\d+$', tag) is not None

def get_entry_type(tag):
    if is_text_tag(tag):
        return "text"
    return "latex"

def baseline_text_to_latex_via_llm(english_of_equation):
    # using chatgpt because local llm took *way* too long (15+ min for one)

    preprompt = "take this transcription and write the equation in latex." + \
    "Don't allow any english and use only latex."

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages = [
            {"role": "system", "content" : preprompt},
            {"role": "user", "content" : english_of_equation}
        ]
    )

    latex_from_llm = completion.choices[0].message.content
    return latex_from_llm.strip()


# loading outside so don't need ot load more than once
print("loading STT model")
speech_to_text_model = whisper.load_model("small") 
def speech_to_text(audio_file_path):
    model = speech_to_text_model
    result = model.transcribe(str(audio_file_path), fp16=False) # might take a bit..
    return result["text"]

def score_one_latex_similarity(l1_text, l2_text):
    return int(l1_text == l2_text)

def text_tag_to_latex_tag(text_tag):
    s = text_tag.split("_")
    s[3] = "latex"
    del(s[4])
    return "_".join(s)

def score_latex_similarity(audiodb, predictions):
    right = 0
    score_data = {}

    for label, latex_prediction in predictions.items():
        real = audiodb[text_tag_to_latex_tag(label)]["data"]
        score = score_one_latex_similarity(real, latex_prediction)
        score_data[label] = {"score": score, "real": real, "pred": latex_prediction}
        right += score

    return right / len(predictions), score_data

    

def baseline_predict_one(audio_file_path):
    text = speech_to_text(audio_file_path)
    latex = baseline_text_to_latex_via_llm(text)
    return latex

def baseline_predict(audiodb):
    preds = {}
    i = 0
    for tag, value in tqdm(audiodb.items()):
        if value["type"] == "text":
            # want to predict
            i += 1
            
            preds[tag] = baseline_predict_one(value["audio_path"])

            if i == 25:
                print("STOPPING EARLY FOR SPEED; remove if needed")
                break

    return preds    

def read_audiodb(audio_folder_path):
    audio_db = {}
    with open(audio_folder_path / "tagged_plain_english_dataset.txt", "r") as eq_file:
        for line in eq_file:
            if len(line) > len("Equation_num_"):
                # line has tag
                tag = line.split(": \"")[0]
                value = line[len(tag)+3:] # +3 to get rid of sepeartor
                entry_type = get_entry_type(tag)
                audio_db[tag] = {"data":value, "type":entry_type}
                if entry_type == "text":
                    audio_db[tag]["audio_path"] = audio_folder_path / (tag+".mp3")
    return audio_db

def baseline_alg_predict(audio_folder_path, preds_json_path):
    audiodb = read_audiodb(audio_folder_path)
    predictions = baseline_predict(audiodb)
    print(f"saving predictions to {preds_json_path}")
    with open(preds_json_path, "w") as json_file:
        json.dump(predictions, json_file)

def baseline_score_predictions(json_predictions_path, audio_folder_path, score_json_path):
    audiodb = read_audiodb(audio_folder_path)
    with open(json_predictions_path, "r") as json_file:
        predictions = json.load(json_file)
    accuracy, score_data = score_latex_similarity(audiodb, predictions)
    print("accuracy:", accuracy)
    print(f"saving score data {score_json_path}")
    with open(score_json_path, "w") as json_file:
        json.dump(score_data, json_file, indent=4)



def main():
    print("Starting predictions.")

    audiodb_folder_path = Path(__file__).parent / "./audiodb"

    preds_path = "predictions.json"
    score_data_path = "score_data.json"

    baseline_alg_predict(audiodb_folder_path, preds_path)
    print("Done predictions.")

    print("Starting scoring.")
    baseline_score_predictions(preds_path, audiodb_folder_path, score_data_path)
    print("Done scoring.")



if __name__ == "__main__":
    main()