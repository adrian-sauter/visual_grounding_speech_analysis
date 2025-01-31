import torch
from collections import OrderedDict
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


import VGBERT_model


def load_pretrained_vg_bert_model(checkpoint_path, device, output_hidden=True):
    # load the language model, image model, and cross-attention model
    language_model = VGBERT_models.Bert_object_hidden(embedding_dim=768, output_hidden=output_hidden).to(device)
    image_model = VGBERT_models.VGG16_Attention(embedding_dim=768, pretrained=True, use_position="learn").to(device)
    cross_attention_model = VGBERT_models.CrossAttention(num_heads=8, subspace_dim=32, embedding_dim=768, relation_base=115).to(device)

    # load transfer model and checkpoint
    transfer_model = VGBERT_models.FullModels.TransferCrossModalRetrieval(image_model, language_model, cross_attention_model)  # TransferCrossModalRetrieval
    state_dict = torch.load(checkpoint_path, map_location=device)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    transfer_model.load_state_dict(new_state_dict,strict=False)
    transfer_model.eval()
    transfer_model.to(device)

    vg_bert = transfer_model.language_model
    image_model = transfer_model.image_model

    return vg_bert

def load_pretrained_bert_model(pretr_type='bert-base-uncased', output_hidden=True, device=None):
    assert device, 'Make sure you specify the device'
    return BertModel.from_pretrained(pretr_type, output_hidden_states=output_hidden).to(device)


def get_layerwise_embeddings(model, tokenizer, language_input, device):
    """
    Get intermediate embeddings from a model for a given language input.

    Args:
        model: The model from which to extract embeddings.
        tokenizer: Tokenizer corresponding to the model.
        language_input: Text input for the model.

    Returns:
        dict: A dictionary where keys are layer names ('layer_0', 'layer_1', etc.) and values are the corresponding embeddings.
    """
    model = model.to(device)

    # Tokenize and encode the input word with special tokens
    encoded_input = tokenizer(language_input, return_tensors='pt', add_special_tokens=True)

    # Move tensors to the same device as the model
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    token_type_ids = encoded_input['token_type_ids'].to(device) if 'token_type_ids' in encoded_input else None

    with torch.no_grad():
        if isinstance(model, BertModel):  # Ungrounded model
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:  # Visually grounded model
            outputs = model.language_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    hidden_states = outputs.hidden_states  # Tuple of tensors (one for each layer)

    # Extract embeddings for the first token (skipping special tokens)
    layer_embeddings = {
        f'layer_{i}': hidden_state[0, 1].cpu().numpy()
        for i, hidden_state in enumerate(hidden_states[1:])  # Skip layer 0 (input embeddings)
    }

    return layer_embeddings

