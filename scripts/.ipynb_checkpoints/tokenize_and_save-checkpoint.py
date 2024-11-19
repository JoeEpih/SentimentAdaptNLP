import pandas as pd
import torch
from transformers import BertTokenizerFast

def tokenize_data(data, max_length=128, batch_size=200):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    all_input_ids = []
    all_attention_masks = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        tokens = tokenizer(
            list(batch['reviewText']),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        all_input_ids.append(tokens['input_ids'])
        all_attention_masks.append(tokens['attention_mask'])

    return {
        'input_ids': torch.cat(all_input_ids),
        'attention_mask': torch.cat(all_attention_masks)
    }

# Load datasets
train = pd.read_csv('train_electronics.csv')
val = pd.read_csv('val_electronics.csv')
test = pd.read_csv('test_movies.csv')

# Tokenize datasets
train_tokens = tokenize_data(train)
val_tokens = tokenize_data(val)
test_tokens = tokenize_data(test)

# Save tokenized datasets
torch.save(train_tokens, 'train_tokens.pt')
torch.save(val_tokens, 'val_tokens.pt')
torch.save(test_tokens, 'test_tokens.pt')

# Save labels
torch.save(torch.tensor(train['sentiment'].tolist()), 'train_labels.pt')
torch.save(torch.tensor(val['sentiment'].tolist()), 'val_labels.pt')
torch.save(torch.tensor(test['sentiment'].tolist()), 'test_labels.pt')
