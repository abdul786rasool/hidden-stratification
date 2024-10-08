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
    texts, y_dict = zip(*batch)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokenized_batch = tokenizer(
        list(texts),  # List of texts
        truncation=True,
        padding=True,  # Perform padding here
        return_tensors='pt'
    )

    return {
        'input_ids': tokenized_batch['input_ids'],
        'token_type_ids':tokenized_batch['token_type_ids'],
        'attention_mask': tokenized_batch['attention_mask'],
        
    }, y_dict

def SentenceTransformer(**kwargs):
    model = SentenceTransformerForClassification(**kwargs)
    return model   

