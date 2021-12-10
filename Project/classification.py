from transformers import pipeline
import torch

if torch.cuda.is_available():
    num = 1
else:
    num = 0



#classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = pipeline("zero-shot-classification", device=0)  # to utilize GPU

topic_labels = {"environment", "politics", "advertisement", "public health", "research", "science", "music",
                "elections", "economics", "sport", "education", "business", "technology", "history", "entertainment"}


def classify_text(input_text):
    sequence = input_text
    candidate_labels = ["environment", "politics", "advertisement", "public health", "research", "science", "music",
                "elections", "economics", "sport", "education", "business", "technology", "history", "entertainment"]
    return classifier(sequence, candidate_labels, multi_class=True)


#result = classify_text("Deep learning is really cool idea", topic_labels)
#print(result['labels'][0], ": ", result['scores'][0])