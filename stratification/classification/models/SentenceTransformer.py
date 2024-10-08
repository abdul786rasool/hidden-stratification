import warnings

# Suppress the FutureWarning for clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*"
)


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login

# Log in using your Hugging Face token
login("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")

class SentenceTransformerForClassification(nn.Module):
    def __init__(self, num_classes):
        super(SentenceTransformerForClassification, self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.activation_layer_name = "module.model.pooler.activation"

    def forward(self, inputs):
        outputs = self.model(**inputs)
        outputs = outputs.pooler_output
        logits = self.classifier(outputs)  # [batch_size, num_classes]
        return logits


def collate_fn_SentenceTransformer(batch):
    # Tokenize each text in the batch
    texts, y_dicts = zip(*batch)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer.clean_up_tokenization_spaces = True
    tokenized_batch = tokenizer(
        list(texts),  # List of texts
        truncation=True,
        padding=True,  # Perform padding here
        return_tensors='pt
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
        'token_type_ids':tokenized_batch['token_type_ids'],
        'attention_mask': tokenized_batch['attention_mask'],
        
    }, merged_y_dict

def SentenceTransformer(**kwargs):
    model = SentenceTransformerForClassification(**kwargs)
    return model   

