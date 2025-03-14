import os
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from utils.mT5_train import training_loop, plot_training
from utils.bleu import calculate_bleu_and_chrf
from peft import get_peft_model, LoraConfig

# Global parameters for translation training
language1 = "en"
language2 = "la"
path_to_train = "/Data/AxelDlv/LatinSummarizer/prompt_no_stanza_train.csv"
path_to_test = "/Data/AxelDlv/LatinSummarizer/prompt_no_stanza_test.csv"
path_to_special_tokens = "/Data/AxelDlv/LatinSummarizer/common_tags_la_en.csv"

max_seq_len = 412
max_new_tokens = 412
model_pretrained = "mt5-small"
model_name = f"/Data/AxelDlv/mt5-small-en-la-translation/{model_pretrained}"  # Adjust as needed
preexisting_checkpoint_path = "/Data/AxelDlv/mt5-small-en-la-translation/mt5-small-en-la-translation-final_no_stanza-epoch-12.pt"  # Leave empty if not using a previous checkpoint
checkpoint_dir = f"/Data/AxelDlv/mt5-small-en-la-translation"
checkpoint_prefix = "mt5-small-en-la-translation-final_no_stanza"

batch_size = 16
lr = 5e-4
start_epoch = 1
end_epoch = 30

os.makedirs(checkpoint_dir, exist_ok=True)
warnings.filterwarnings("ignore")

print(f"Aiming to train a model to translate and summarize {language2} texts using {language1} as a pivot.")
print(f"Training data: {path_to_train} | Testing data: {path_to_test}")
print(f"Training for {end_epoch} epochs with batch size {batch_size} and learning rate {lr}.")

try:
    df_train = pd.read_csv(path_to_train)
    df_test = pd.read_csv(path_to_test)
except:
    # Give an example of df_train and df_test
    df_train = pd.DataFrame({
        "prefix": ["<en.la>", "<en.la>"],
        "prompt": ["This is a sample text", "This is another sample text"],
        "answer": ["Hic est exemplum", "Hic est aliud exemplum"]
    })
    df_test = pd.DataFrame({
        "prefix": ["<en.la>", "<en.la>"],
        "prompt": ["This is a sample text", "This is another sample text"],
        "answer": ["Hic est exemplum", "Hic est aliud exemplum"]
    })

# Load special tokens and initialize tokenizer
special_tokens = pd.read_csv(path_to_special_tokens)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens['token'].tolist()})

# Load the model and adjust token embeddings
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
model.resize_token_embeddings(len(tokenizer))

# Integrate LoRA using PEFT (adjust target modules as needed)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

if preexisting_checkpoint_path and os.path.exists(preexisting_checkpoint_path):
    model.load_state_dict(torch.load(preexisting_checkpoint_path))
    print(f"Loaded checkpoint from: {preexisting_checkpoint_path}")
    
print("LoRA integration complete.")

print(f"Special tokens added: {special_tokens['token'].tolist()}")
print(f"Model parameters: {model.num_parameters():,}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}, total tokens: {len(tokenizer)}")

# Setup optimizer and scheduler
n_batches = int(np.ceil(len(df_train) / batch_size))
total_steps = n_batches * (end_epoch - start_epoch + 1)
n_warmup_steps = int(total_steps * 0.01)

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=total_steps)

print("Tokenizer length:", len(tokenizer)) 

vocab_size_in_model = model.get_input_embeddings().weight.shape[0]
print("Model embedding size:", vocab_size_in_model)

# Run the training loop
losses = training_loop(
    model, 
    df_train, 
    df_test, 
    tokenizer, 
    optimizer, 
    scheduler,
    start_epoch, 
    end_epoch, 
    batch_size,
    checkpoint_dir, 
    checkpoint_prefix,
    print_freq=1, 
    max_seq_len=max_seq_len,
    use_amp=False, 
    accumulation_steps=1,
    column_prompt="prompt", 
    column_target="answer",
    fraction_to_use=0.15,
)