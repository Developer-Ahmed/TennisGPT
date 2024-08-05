import torch
import torch.nn as nn
from torch.nn import functional as F
import math
# initialize hyperparamaters
from config import ConfigClass as config

class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # initiliaze all three attributes (key, value, query) in one layer
        self.attributes = nn.Linear(config.num_embed, config.num_embed * 3, bias=config.bias)
        # residual projection
        self.ln_output_layer = nn.Linear(config.num_embed, config.num_embed, bias=config.bias)
        self.dropout_value = config.dropout
        self.dropout = nn.Dropout(self.dropout_value)
        self.num_head = config.num_head

    def forward(self, x):
        batch, seq, embed = x.shape
        attributes = self.attributes(x)

        attr_embed_size = attributes.shape[2]
        attributes = attributes.view(batch, seq, self.num_head, attr_embed_size // self.num_head).transpose(1, 2)
        # split the embedding dimension evenly to get query, key, and value
        query, key, value  = attributes.split(attributes.shape[3] // 3, dim=3)

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_value,
            is_causal=True
        )
        
        # reshape output to feed into output layer
        output = output.contiguous().view(batch, seq, embed)
        final_output = self.dropout(self.ln_output_layer(output))
        return final_output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # filters applied is four times the input
        self.linear_1 = nn.Linear(config.num_embed, config.num_embed * config.filter_size, bias=config.bias)
        # residual projection
        self.output_layer = nn.Linear(config.num_embed * config.filter_size, config.num_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = self.linear_1(x)
        output = self.gelu(output)
        output = self.output_layer(output)
        output = self.dropout(output)
        return output

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(config.num_embed, bias=config.bias)
        self.attention = FlashAttention(config)
        self.layernorm_2 = nn.LayerNorm(config.num_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # add x for residual connections
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            word_embed = nn.Embedding(config.vocab_size, config.num_embed),
            pos_embed = nn.Embedding(config.block_size, config.num_embed),
            dropout = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layer)]),
            layer_norm = nn.LayerNorm(config.num_embed, bias=config.bias),
        ))
        # ??? bias: true
        self.linear_head = nn.Linear(config.num_embed, config.vocab_size, bias=config.bias)

        # weight tying
        self.transformer.word_embed.weight = self.linear_head.weight

        # initiliaze weights in normal distribution
        self.apply(self._init_weights)

        # nummber of parameters in millions
        total_parameters= self.get_num_params()/1e6
        print(f"Number of Parameters: {total_parameters:.2f}M")

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # special init for residual projections
        for name, parameter in module.named_parameters():
            if "output_layer.weight" in name:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layer))

    def forward(self, x):
        B, T = x.shape

        # forward model
        token_embed = self.transformer.word_embed(x) # shape (batch, seq, embed)
        positions = torch.arange(0, T, dtype=torch.long)
        pos_emb = self.transformer.pos_embed(positions) # shape (seq, embed)
        hidden_states = token_embed + pos_emb # shape (batch, seq, embed)
        hidden_states = self.transformer.dropout(hidden_states) # same shape
        for block in self.transformer.blocks: # same shape
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.layer_norm(hidden_states) # same shape
        logits = self.linear_head(hidden_states) # shape (batch, seq, vocab_size)

        return logits # shape (batch, seq, vocab_size)
