import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login

# Log in using your Hugging Face token
login("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")



class LlamaForClassification(nn.Module):
    def __init__(self, num_classes):
        super(LlamaForClassification, self).__init__()
        self.llama = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.classifier = nn.Linear(self.llama.config.hidden_size, num_classes)
        self.dummy_layer = nn.Identity()
        self.activation_layer_name = "dummy_layer"

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask=inputs['attention_mask']
        outputs = self.llama.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  
        sequence_lengths = attention_mask.sum(dim=1) - 1  # Subtract 1 to get the last non-padding token index

        # Extract the hidden states corresponding to the last non-padding token for each sequence
        cls_tokens = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]

        # Save the CLS tokens as the embedding output (named as the custom layer)
        self.dummy_layer(cls_tokens)

        # Pass the `[CLS]` representation through the classifier to get logits for each class
        logits = self.classifier(cls_tokens)  # [batch_size, num_classes]
        return logits


def collate_fn_llama2(batch,y_dict):
    # Tokenize each text in the batch
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_batch = tokenizer(
        batch,  # List of texts
        truncation=True,
        padding=True,  # Perform padding here
        return_tensors='pt'
    )

    return {
        'input_ids': tokenized_batch['input_ids'],
        'attention_mask': tokenized_batch['attention_mask']
    }, y_dict


def llama2(**kwargs):
    model = LlamaForClassification(**kwargs)
    return model   

