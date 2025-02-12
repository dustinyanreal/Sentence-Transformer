import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MODEL = AutoModel.from_pretrained("bert-base-uncased")

class SentenceTransformer(nn.Module):
    def __init__(self,
                 model=MODEL,
                 tokenizer=TOKENIZER,
                 num_classes=0,
                 num_sentiments=0):
        super(SentenceTransformer, self).__init__()
        self.model = MODEL
        self.tokenizer = TOKENIZER
        self.num_classes = num_classes
        self.num_sentiments = num_sentiments
        self.dropout = nn.Dropout(0.1)

        self.topic_head = nn.Linear(self.model.config.hidden_size, self.num_classes)

        self.sentiment_head = nn.Linear(self.model.config.hidden_size, self.num_sentiments)

    def forward(self, inputs):
        output = self.model(**inputs)
        mean_output = output.last_hidden_state.mean(dim=1)
        sentence_embedding = F.normalize(mean_output, p=2, dim=1)
        mean_output = self.dropout(mean_output)
        
        topic_logits = self.topic_head(mean_output)

        sentiment_logits = self.sentiment_head(mean_output)

        return sentence_embedding, topic_logits, sentiment_logits

    def encode(self, sentence):
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
            sentence_embeddings, _, _ = self.forward(inputs)
            return sentence_embeddings

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
                    "topic": predicted_topic.item(),
                    "sentiment": predicted_sentiment.item()
                    }

