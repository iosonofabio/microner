# Example from: https://medium.com/@ahmetmnirkocaman/mastering-named-entity-recognition-with-bert-a-comprehensive-guide-b49f620e50b0
# Import necessary libraries
import numpy as np
import torch
from transformers import MobileBertTokenizer, MobileBertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from keras_preprocessing.sequence import pad_sequences

print('Prepare training')
training_table = [
    ['Scatter age against expression', ['B-PLOTYPE', 'B-AXIS1', 'O', 'B-AXIS2']],
]
sentences, labels = list(zip(*training_table))

# tag id dictionaries
unique_tags = set(tag for doc in labels for tag in doc) | set(["PAD"])
for tag in frozenset(unique_tags):
    if tag.startswith('B-'):
        unique_tags |= set(['I' + tag[1:]])
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

print('Fix a bunch of constants')
NUM_LABELS = len(unique_tags)
MAX_LEN = 10
EPOCHS = 2

print('Load the tokenizer and model')
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertForTokenClassification.from_pretrained("google/mobilebert-uncased", num_labels=NUM_LABELS)  # Set NUM_LABELS according to your dataset

print('Tokenize')
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence.split(' '), text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        # NOTE: This is a homemade if statment, looks intuitively correct but is it?
        if (n_subwords > 1) and label.startswith('B-'):
            ilabel = 'I' + label[1:]
            labels.extend([label] + [ilabel] * (n_subwords - 1))
        else:
            labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

# Assuming `sentences` and `labels` are lists containing sentences and their corresponding labels
tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

print('Convert tokenized text and labels to tensor format using standard function of keras_preprocessing')
input_ids = pad_sequences(
    [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post",
)
tags = pad_sequences(
    [[tag2id.get(l) for l in lab] for lab in labels],
    maxlen=MAX_LEN, dtype="long", value=tag2id["PAD"], truncating="post", padding="post",
)

# Create attention masks to ignore padded tokens
attention_masks = np.array([[float(i != 0.0) for i in ii] for ii in input_ids]).astype(np.float32)

print('Create a pytorch Dataset')
class CustomDataset(Dataset):
    def __init__(self, input_ids, tags, attention_masks):
        self.input_ids = input_ids
        self.tags = tags
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.tags[idx], self.attention_masks[idx]


# Split data into train and validation sets and wrap them in DataLoader for efficient batching
training_data = CustomDataset(input_ids, tags, attention_masks)  # NOTE: using everything for training for now :-)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)#

print('Fine-tuning setup')
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print('Train')
for epoch in range(1, EPOCHS + 1):
    print(f'Epoch: {epoch}')
    model.train()
    for batch in train_dataloader:
        # NOTE: I think this unpacking corresponds to the custom Dataset __getitem__ function?
        b_input_ids, b_labels, b_masks = batch
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    # Validation loop can be added here

#print('Store fine-tuned model')
#model.save_pretrained("./ner_model")
