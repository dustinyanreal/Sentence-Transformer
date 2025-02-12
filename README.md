# Sentence Transformer

## Overview

This project processes sentences to generate embeddings for various NLP tasks. These embeddings are then used for sentence classification and sentiment analysis, enabling deeper insights into textual data.


## Task 1: Sentence Transformer Implementation

In Task1, I built a class that servers as my transformer mode, leveraging **bert-base-cased** as the backbone. 

I chose this pre-trained BERT model because it has been trained on a vast corpus of text, allowing it to capture contextual representations of words. Using this pre-trained model significantly recudes training time and computational costs while also providing good performance for many NLP tasks. Additionally it comes with a pre-trained tokenize, which makes consistent tokenization with the model's expectaions.

To generate sentence embeddings, I developed an encode function that transforms a regular sentence into a numerical represenation.
- The first step in this process is tokenization, breaking down a sentence into individual tokens. Since **bert-base-cased** requires 512 tokens as an input, I normallized each sentence by padding shorter ones and truncating longer ones. This ensurres that all input sequences maintain a uniform lenth.
- Once tokenized, I produced sentence embeddings by taking the mean of the last hidden states. This helps capture semantic meaning of a sentence.
- Finally, I normalized the sentence embeddings to ensure that their values remain in a consistent range. Normalization helps improve numerical stability and enhances performance in tasks. (i.e classification and sentiment analysis **TASK 2)

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

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MODEL = AutoModel.from_pretrained("bert-base-uncased")

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
