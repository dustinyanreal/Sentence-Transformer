# Sentence Transformer

## Overview

This project processes sentences to generate embeddings for various NLP tasks. These embeddings are then used for sentence classification and sentiment analysis, enabling deeper insights into textual data.


## Task 1: Sentence Transformer Implementation

In Task 1, I built a class that serves as my transformer mode, leveraging **bert-base-cased** as the backbone. 

I chose this pre-trained BERT model because it has been trained on a vast corpus of text, allowing it to capture contextual representations of words. Using this pre-trained model significantly reduces training time and computational costs while also providing good performance for many NLP tasks. Additionally it comes with a pre-trained tokenizer, which makes consistent tokenization with the model's expectaions.

To generate sentence embeddings, I developed an encode function that transforms a regular sentence into a numerical represenation.
- The first step in this process is tokenization, breaking down a sentence into individual tokens. Since **bert-base-cased** requires 512 tokens as an input, I normallized each sentence by padding shorter ones and truncating longer ones. This ensurres that all input sequences maintain a uniform lenth.
- Once tokenized, I produced sentence embeddings by taking the mean of the last hidden states. This helps capture semantic meaning of a sentence.
- Finally, I normalized the sentence embeddings to ensure that their values remain in a consistent range. Normalization helps improve numerical stability and enhances performance in tasks. (i.e classification and sentiment analysis **TASK 2**)

**Reason for my building structure:**

- **Transformer Backbone:** ``` bert-base-cased ``` has been trained on a massive corpus, enabling it to generate high-quality embeddings. The cased variant captures capitalization which is crucial in Task 2, where capitalization can be the different in topics and sentiment analysis.
- **AutoTokenizer:** This ensured that the tokenization process was the same in how BERT was trained. If I used another tokenizer such as NLTK, it could lead to embeddings that give suboptimal performance.
- **Mean Pooling:** While there are pooling methods, such as maxpooling which focuses on the most activated token, mean pooling gives more of a balanced representation of a sentence.
- **Normalization:** This method prevents large variations, by keeping embeddings in a similar numerical range.


**Code for Task 1**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
MODEL = AutoModel.from_pretrained("bert-base-cased")

class SentenceTransformer(nn.Module):
    def __init__(self, model=MODEL, tokenizer=TOKENIZER)
    super(SentenceTransformer, self).__init__()
    
    self.model = MODEL
    self.tokenizer = TOKENIZER


    def encode(self, sentence):
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
            output = self.model(**inputs)
            sentence_embeddings = output.last_hidden_state.mean(dim=1)
            return F.normalize(sentence_embeddings, p=2, dim=1)
```

## Task 2: Multi-Task Learning Expansion

For Task 2, I expanded my ``` class SentenceTransformer(nn.Module) ``` to be able to do Sentence Classification and Sentiment Analysis.

When initalizing, I added the number of topics, number of sentiments, and 2 heads for predicting the topic and analyzing the sentiment.
```python

def __init__(self,
             model = MODEL,
             tokenizer = TOKENIZER,
             num_classes=0 # *NEW* used for the number of topics to be found. Initialized to 0 if nothing is found
             num_sentiments=0 # *NEW* used for the number of different sentiments. Initiliazed to 0 if nothing is found

    super(SentenceTransformer, self).__init__()
    self.model = MODEL
    self.tokenizer = TOKENIZER
    self.num_classes = num_classes
    self.num_sentiments = num_sentiments
    
    #Topic head is used to predict the topic of a given sentence.
    self.topic_head = nn.Linear(self.model.config.hidden_size, self.num_classes) #NEW
    #Sentiment head is used to analyze the sentiment of a given sentence.
    self.topic_head = nn.Linear(self.model.config.hidden_size, self.num_sentiments #NEW
```

Along with adding new elements to the initalization phase, I also added two functions, **forward** and **predict**.
``` def forward(self, inputs) ``` is essentially the same as encode, but can be used to input embeddings into different heads to return a different output to a given NLP task.
```python
def forward(self, inputs):
    output = self.model(**inputs)
    mean_output = out.last_hidden_state.mean(dim=1)
    sentence_embedding = F.normalize(mean_output, p=2, dim=1)
    
    topic_logits = self.topic_head(sentence_embedding)
    sentiment_logits = self.sentiment_head(sentence_embedding)

    return sentence_embedding, topic_logits, sentiment_logits
```

``` def predict(self, sentence) ``` is used like the function states, it predicts the outcome of a sentence topic and analyzes the sentiment of a sentence.
```python
def predict(self, sentence):
    self.eval()
    with torch.no_grad():
        inputs = self.tokenizer(
                sentence,
                padding=True,
                truncation=True,
                return_tensors="pt"
                )
        _, topic_logits, sentiment_logits = self.forward(inputs)
        topic_probs = torch.softmax(topic_logits, dim=1)
        predicted_topic = torch.argmax(topic_probs, dim=1)

        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        predicted_sentiment = torch.argmax(sentiment_probs, dim=1)

        return {
                "topic: predicted_topic.item(),
                "sentiment": predicted_sentiment.item()
                }
```
