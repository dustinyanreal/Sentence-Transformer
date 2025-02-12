import sys
import os
import json
from transformer_cls import SentenceTransformer

def read_file():
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        else:
            print("Not a valid file")
    else:
        return {
                "topic_labels": [],
                "sentiment_labels": [],
                "sentences": []
                }

def print_results(sentence, inputs, model):
    embedding = model.encode(sentence)

    print(f"Input Sentence: {sentence}")

    prediction = model.predict(sentence)
    print(f"Embedding: {embedding}")
    print(f"Task A - Topic: {inputs['topic_labels'][prediction['topic']]}")
    print(f"Task B - Sentiment Analysis: {inputs['sentiment_labels'][prediction['sentiment']]}")



def main():
    inputs = read_file()
    if len(inputs["topic_labels"]) == 0:
        inputs['topic_labels'] = ["Food", "Comedy", "Technology", "Sports"]
    if len(inputs["sentiment_labels"]) == 0:
        inputs['sentiment_labels'] = ["Positive", "Neutral", "Negative"]

    model = SentenceTransformer(
            num_classes=len(inputs['topic_labels']),
            num_sentiments=len(inputs['sentiment_labels'])
            )

    if len(inputs["sentences"]) == 0:
        while True:
            sentence = input("Enter a sentence (type 'exit' or 'q' to quit): ")

            if sentence.lower() == "exit" or sentence.lower() == "q":
                break
            else:
                print_results(sentence, inputs, model)
    else:
        for sentence in inputs["sentences"]:
            print_results(sentence, inputs, model)

if __name__ == "__main__":
    main()
