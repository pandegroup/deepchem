from transformers import BertTokenizer
import torch

# Load the pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input smiles_batch
inputs = tokenizer(smiles_batch[0].tolist(), padding=True, return_tensors='pt', truncation=True)

# Generate attention masks
attention_masks = torch.ones(inputs['input_ids'].shape)
attention_masks[inputs['input_ids'] == tokenizer.pad_token_id] = 0
