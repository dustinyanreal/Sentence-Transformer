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

For Task 2, I extended my ``` SentenceTransformer(nn.Module)``` class to support sentence classification and sentiment analysis. This allows the model to not only generate sentence embeddings but also perform other NLP tasks.


### Initlization

In the ``` __init__``` method, I added two new parameters:
- ```num_classes```: this represents the number of possible topics in sentence classification. Defaults to ``` 0``` if unspecified.
- ``num_sentiments```: this represents the number of different sentiment categories. Defaults to ``` 0``` if unspecified.

I also added two separate linear layers as classification heads:
- ```self.topic_head```: this predicts the topic of a given sentence
- ```self.sentiment_head```: this analyzes the sentiment of a given sentence
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
This design ensures that if no topics or sentiments are provided, the model still functions as a sentence embedder.

### Adding forward and predict Functions

I introduced two new functions:
- ```forward```: handles computation, generating sentence embeddings and passing them through the different classification heads
- ```predict```: converts embeddings, inference through ```forward```, applies softmax for probability distribution, and determines the most likely topic and sentiment.

### forward Function

The forward function takes tokenized inputs and generates sentence embeddings. These embeddings are then fed into classification heads for topic prediciton and sentiment analysis.

```python
def forward(self, inputs):
    output = self.model(**inputs)
    mean_output = out.last_hidden_state.mean(dim=1)
    sentence_embedding = F.normalize(mean_output, p=2, dim=1)
    
    topic_logits = self.topic_head(sentence_embedding)
    sentiment_logits = self.sentiment_head(sentence_embedding)

    return sentence_embedding, topic_logits, sentiment_logits
```

The implementation is similar to the ```encode``` function in task one, where instead of just returning the sentence embedding when I normalize it, I pass the normalized embedding into the different classification heads, ```topic_head``` and ```sentiment_head```.

### predict Function

The predict function provides inference on raw text. It tokenizes the sentence, passes it through ```forward```, applies softmax to convert logits into probabilities, and extracts the most probable class using argmax.

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

- Softmax is used to convert logits into probabilities that sum up to 1. The higher the softmax probability, the more confident the model is to predict if a sentence is under a specific topic.
- Argmax, selects the class with the highest confidence, just finds the max index.
