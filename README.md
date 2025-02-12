#Sentence Transformer

##Overview
This is a project that takes in sentences and analyzes them. First the project produces embeddings used for different NLP tasks. Next the project takes those embeddings and uses them in Sentence Classification and Sentiement Analysis.

##Task 1
In Task 1, I built a class that acts as my transformer model.

The transformer backbone is 'bert-base-uncased'. I chose to use a pre-trained bert model because... This also came with a pre-trained tokenizer ...

I developed a function, encode to transform a regular sentence into a sentence embedding.
I took a sentence and tokenized it. That means for every word inside a sentence, it became a token. This also meant that I had to normalize every sentence. If a sentence was too short, I had to pad it. If a sentence was too long, I had to truncate it so that every sentence was the same length. I had to do this so that all inputs entering the model would be the same.

Then I produced the sentence embeddings with the last hidden state's mean... (reason)

Lastly, I normalized the sentence embeddings. (reason)

**Reason for my building structure:** (reasons for choices)

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
