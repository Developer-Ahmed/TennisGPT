import torch
import torch.nn as nn
from torch.nn import functional as F
# initialize hyperparamaters
from config import ConfigClass as config
from model import GPT
from load_data import tokenizer
import os

finetuned_path = config.finetuned_model_path
if not os.path.exists(finetuned_path):
    raise FileNotFoundError(f"The file path {finetuned_path} does not exist. Please finetune first.")

model = GPT(config)
model.load_state_dict(torch.load(finetuned_path))

def generate_answer(context):
    """
    Context is a string containing user input, so we need process and tokenize it.
    """
    tokenized_text = tokenizer.encode_ordinary(context)
    # add padding if context is smaller than block size
    if len(tokenized_text) < config.block_size:
        num_padding = config.block_size - len(tokenized_text)
        pad_id = 50258
        tokenized_text = tokenized_text + [pad_id] * num_padding
    # if greater then crop it to block size
    elif len(tokenized_text) > config.block_size:
        tokenized_text = tokenized_text[-config.block_size:]
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.int32)
    # add batch dimension
    tokenized_text = tokenized_text.view(1, -1)

    answer = []
    with torch.no_grad():
        while True:
            predictions = model(tokenized_text)
            batch_classes = predictions[:, -1, :] # get rid of seq length
            # normalize predictions to get probabilities 
            probabilities = F.softmax(batch_classes, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # check if model's response is finished
            if next_token.item() == tokenizer.eot_token:
                break
            
            answer.append(next_token.item())
            # remove the first token and add this token to the context
            tokenized_text = torch.cat((tokenized_text, next_token), dim=1)
            tokenized_text = tokenized_text[:, -config.block_size:]

    return tokenizer.decode(answer)

# chat loop
print("-------------------")
print("Chat with our model! Type q to quit.")
print("-------------------")
while True:
    user_input = input("YOU: ")
    if user_input == "q":
        break
    model_answer = generate_answer(user_input)
    print("MODEL: ", model_answer)
