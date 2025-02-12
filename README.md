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

For Task 2, I extended my ```SentenceTransformer(nn.Module)``` class to support sentence classification and sentiment analysis. This allows the model to not only generate sentence embeddings but also perform other NLP tasks.


### Initlization

In the ```__init__``` method, I added two new parameters:
- ```num_classes```: this represents the number of possible topics in sentence classification. Defaults to ```0``` if unspecified.
- ```num_sentiments```: this represents the number of different sentiment categories. Defaults to ```0``` if unspecified.

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
    self.topic_head = nn.Linear(self.model.config.hidden_size, self.num_sentiments) #NEW
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

## Task 3: Training Considerations

### Freeze Entire Network

**Description:** All network parameters are frozen.

**Implications:**
- Model will not learn, weights will not update.
- The model relies solely on the pre-trained representation

**Advantages:**
- If the task is similar to the pre-trained task, the model will be able to perform the task well.
- Doesn't require intensive GPUs usage since gradients aren't being computed.

**When to train:**
- Freezing entire network makes the model non-trainable. This should only be use if the task does not require additional training from the base model, i.e., just for inference.
- If the task closely resembles what the model was pre-trained on, the model should perform sufficiently without further training.

### Freeze Transformer Backbone

**Description:** The backbone layers (e.g., BERT) are frozen.

**Implications:**
- The heads can be trained to adapt to specific tasks, such as Task A (Sentence Topic Classification) and Task B (Sentiment Analysis).
- Training time is reduced since the backbone’s weights remain unchanged. The focus shifts to training the heads.


**Advantages:**
- Can make the head of the model specialize in a specific task without losing pre-training representations.
- The head can specialize in a specific task without losing the benefits of the pre-trained representations.

**When to train:**
- When you have a downstream task and the model was pre-trained well. For example, if you need to perform sentiment analysis, BERT or GPT is already well-trained, and you only need to train the head for your specific task.

### Freeze Task-Specifc Head

**Description:** One task-specifc head is frozen.

**Implications:**
- Freezing one head allows the other head to train while preserving the features of the frozen head.
- While one head is learning, it does not interfere with the performance of the other head.

**Advantages:**
- This is useful when tasks share a lot of the same information (e.g., Topic Classification & Sentiment Analysis), but one task (e.g., Sentiment Analysis) is performing poorly. Freezing the better-performing head allows more focus on improving the underperforming task.
- Prevents the frozen head from overfitting, while the unfrozen head can adapt to its specific task.

**When to train:**
- When training for more than two tasks. If one task performs well and another requires more optimization, you can freeze the better-optimized head and train the poorly optimized head without affecting the other task.

### Transfer Learning Process

Considering the scenario where transfer learning can be beneficial for this specific project, I will outline the following:
- 1: The choice of a pre-trained model
- 2: The layers I would freeze/unfreeze
- 3: The rational behind these choices

**Choice of a pre-trained model:**

When selecting a pre-trained model, I would choose one that has already been trained on a similar task. For example, in this project, where I am creating a sentence transformer to generate sentence embeddings for topic classification and sentiment analysis, a model like ```bert-base-cased``` would be ideal. BERT has been trained on a large corpus of text, learning general semantics, syntax, and some domain-specific nuances. With a small amount of labeled data, I can fine-tune the model to perform the specific tasks effectively.

If the task were domain-specific, such as analyzing legal documents, I would use a model like LegalBERT, which has been specifically trained on legal texts.

The choice of the pre-trained model depends on the task. In my case, since the tasks are general NLP tasks (topic classification and sentiment analysis), a general NLP model like BERT is suitable.

**The Layers I would freeze/unfreeze**

- **Freeze Entire Backbone:** If the pre-trained features are highly aligned with the task, I would freeze the entire backbone to leverage the pre-trained representations without additional training. This is beneficial if I don't need the model to adapt to new tasks and want to use the pre-trained knowledge directly.

- **Unfreeze Top Layers:** If I want the model to improve performance on a specific task, unfreezing the top layers will allow the model to adapt to the dataset without losing the benefits of pre-trained knowledge. This helps capture task-specific semantics while retaining the general knowledge learned during pre-training.

- **Unfreeze Entire Model: If I have a large amount of data, time, and resources, unfreezing the entire model would allow the model to fully adapt to the new task and domain. However, for most cases, this is not necessary.

In my scenario, where the project involves creating a sentence transformer for topic classification and sentiment analysis, unfreezing the top layers would be most beneficial. Since BERT is a general-purpose NLP model, it is not directly optimized for my tasks. Given my limited data and resources, I cannot afford to fully fine-tune the entire model.

Unfreezing the top layers will enable me to benefit from the pre-trained knowledge in BERT while adapting the model to my specific tasks through the top layers. This allows for a balance between using BERT’s pre-trained knowledge and customizing the model to perform well on topic classification and sentiment analysis.
