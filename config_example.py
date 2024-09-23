from dataclasses import dataclass

@dataclass
class ConfigClass:
    block_size: int = 512
    # GPT-2 tokenizer's vocab_size of 50257 is padded up to nearest multiple of 64 for improved efficiency
    vocab_size: int = 50304 
    num_layer: int = 12
    num_head: int = 12
    num_embed: int = 768
    dropout: float = 0.1
    bias: bool = True # bias in linears and layernorms
    learning_rate: float = 2e-5
    num_epochs: int = 5
    batch_size: int = 16
    # where to save the model after pretraining
    pretrained_model_path: str = "./gpt_pretrained.pth"
    filter_size: int = 4 # used in mlp, 4 times the num_embed
    dataset_name: str = "openwebtext"
    # num_batches for loading into memmap
    loading_batches: int = 16
    compile: bool = True
    pretraining: bool = True