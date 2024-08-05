import torch
import torch.nn as nn
from torch.nn import functional as F
# initialize hyperparamaters
from config import ConfigClass as config
from model import GPT
from load_data import train_loader, val_loader, tokenizer

model = GPT(config)
# initialize with pretrained model's weights for finetuning
if not config.pretraining:
    model.load_state_dict(torch.load(config.pretrained_model_path))

def calc_val_loss():
    val_loss_total = 0
    batches_in_val = 0

    # no backpropagation
    with torch.no_grad():
        model.eval()
        # run on validation set
        for batch_id, data in enumerate(val_loader):
          # end the epoch if there isn't enough tokens for the batch
          if len(data) < (config.batch_size * config.block_size):
            break

          inputs = data[:-1].view(config.batch_size, config.block_size).to(torch.int32)
          targets = data[1:].view(config.batch_size, config.block_size).to(torch.int32)

          # forward pass
          raw_predictions = model(inputs)
          # batch * seq for better efficiency
          flattened_predictions = raw_predictions.view(raw_predictions.shape[0] * raw_predictions.shape[1], raw_predictions.shape[2]).to(torch.float32)
          flattened_targets = targets.view(targets.shape[0] * targets.shape[1]).to(torch.long)
          # any padding ? use ignore_index
          loss = F.cross_entropy(flattened_predictions, flattened_targets)

          # track stats
          val_loss_total += loss.item()
          batches_in_val += 1

    val_loss_average = val_loss_total / batches_in_val
    return val_loss_average

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
if config.compile:
    model = torch.compile(model)

# print initial loss
val_loss = calc_val_loss()
print(f"EPOCH 0: Validation Loss: {val_loss}")

total_steps = 0
for epoch in range(config.num_epochs):
  total_loss = 0
  batches_in_epoch = 0

  for batch_id, data in enumerate(train_loader):
    # end the epoch if there isn't enough tokens for the batch
    if len(data) < (config.batch_size * config.block_size):
      break

    inputs = data[:-1].view(config.batch_size, config.block_size).to(torch.int32)
    targets = data[1:].view(config.batch_size, config.block_size).to(torch.int32)

    # forward pass
    raw_predictions = model(inputs)
    # batch * seq for better efficiency
    flattened_predictions = raw_predictions.view(raw_predictions.shape[0] * raw_predictions.shape[1], raw_predictions.shape[2]).to(torch.float32)
    flattened_targets = targets.view(targets.shape[0] * targets.shape[1]).to(torch.long)
    # any padding ? use ignore_index
    loss = F.cross_entropy(flattened_predictions, flattened_targets)

    # backward pass
    loss.backward()

    # step and clear gradients
    optimizer.step()
    optimizer.zero_grad()

    # track stats
    total_loss += loss.item()
    batches_in_epoch += 1
    total_steps += 1

    print(f"Step {total_steps}: Training Loss:", total_loss / batches_in_epoch)

  loss_average = total_loss / batches_in_epoch

  val_loss_average = calc_val_loss()

  print(f"EPOCH {epoch}: Training Loss: {loss_average} Validation Loss: {val_loss_average}")

if config.pretraining:
    torch.save(model.state_dict(), config.pretrained_model_path)
else:
    torch.save(model.state_dict(), config.finetuned_model_path)