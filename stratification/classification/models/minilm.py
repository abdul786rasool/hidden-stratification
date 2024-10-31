import warnings

# Suppress the FutureWarning for clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*"
)

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

# Log in using your Hugging Face token
login("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")



class MiniLM(nn.Module):
    def __init__(self, num_classes):
        super(MiniLM, self).__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained('microsoft/MiniLM-L12-H384-uncased', num_labels=num_classes)
        self.activation_layer_name = "classifier.bert.pooler.activation"

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        logits = self.classifier(**inputs).logits
        return logits


def collate_fn_miniLM(batch):
    # Tokenize each text in the batch
    texts, y_dicts = zip(*batch)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
    #tokenizer.pad_token = tokenizer.eos_token
    tokenized_batch = tokenizer(
        list(texts),  # List of texts
        max_length=512,
        truncation=True,
        padding=True,  # Perform padding here
        return_tensors='pt'
    )
    merged_y_dict = {}
    for y_dict in y_dicts:
        for key, value in y_dict.items():
            if key not in merged_y_dict:
                merged_y_dict[key] = []
            merged_y_dict[key].append(value.item())    
    for key,value in y_dict.items():
        merged_y_dict[key] = torch.tensor(merged_y_dict[key])
    

    return {
        'input_ids': tokenized_batch['input_ids'],
        'attention_mask': tokenized_batch['attention_mask']
    }, merged_y_dict


def miniLM(**kwargs):
    model = MiniLM(**kwargs)
    return model   

