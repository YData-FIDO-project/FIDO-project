"""
Initializing BERT model
"""

import torch
from transformers import BertModel, BertTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)

        return logits


# creating dataset
class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label_digit'].values
        self.label_name = df['label'].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text,
                                  return_tensors='pt',
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True)

        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)
                }


# creating dataloader (???)
