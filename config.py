from dataclasses import dataclass

@dataclass
class ConfigClass:
    block_size: int = 8
    # GPT-2 tokenizer's vocab_size of 50257 + pad token is padded up to nearest multiple of 64 for improved efficiency
    vocab_size: int = 50304 
    num_layer: int = 2
    num_head: int = 2
    num_embed: int = 4
    dropout: float = 0.0
    bias: bool = True # bias in linears and layernorms
    learning_rate: float = 3e-4
    num_epochs: int = 1
    batch_size: int = 256
    # where to save the model after pretraining
    pretrained_model_path: str = "./saved/gpt_pretrained.pth"
    # where to save the model after finetuning
    finetuned_model_path: str = "./saved/gpt_finetuned.pth"
    filter_size: int = 2 # used in mlp
    huggingface_dataset: bool = False
    dataset_name: str = "tennis"
    # where is the csv file located
    dataset_location: str = "./data/finetuning"
    # num_batches for loading into memmap
    loading_batches: int = 4
    compile: bool = False
    pretraining: bool = False