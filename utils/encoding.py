import torch

# -------------------------
# Data Encoding Functions
# -------------------------
def encode_str(text, tokenizer, seq_len):
    """
    Tokenize, pad/truncate to seq_len, and return (input_ids, attention_mask).
    """
    tokenized = tokenizer(
        text,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]
    return input_ids, attention_mask

def encode_row(row, tokenizer, seq_len, column_prompt="prompt", column_target="answer"):
    """
    Encode input and target text from a single row.
    Returns (input_ids, input_attention, target_ids).
    Assumes row has keys 'input_text' and 'target_text'. Adjust if needed.
    """
    input_text = row[column_prompt]
    target_text = row[column_target]
    
    if not input_text or not target_text:
        return None

    input_ids, input_attention = encode_str(input_text, tokenizer, seq_len)
    target_ids, _ = encode_str(target_text, tokenizer, seq_len)
    return (input_ids, input_attention, target_ids)

def encode_batch(batch, tokenizer, max_seq_len=512, column_prompt="prompt", column_target="answer"):
    """
    Encode a batch of rows into tensors for inputs, attention masks, and targets.
    """
    inputs, masks, targets = [], [], []
    for _, row in batch.iterrows():
        encoded = encode_row(row, tokenizer, max_seq_len, column_prompt, column_target)
        if encoded is None:
            continue
        input_ids, input_attn, target_ids = encoded
        inputs.append(input_ids.unsqueeze(0))
        masks.append(input_attn.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))
    
    if not inputs or not targets:
        return None

    batch_input_ids = torch.cat(inputs, dim=0).cuda()
    batch_attention_mask = torch.cat(masks, dim=0).cuda()
    batch_target_ids = torch.cat(targets, dim=0).cuda()
    return (batch_input_ids, batch_attention_mask, batch_target_ids)

# def data_generator(dataset, tokenizer, batch_size=32, max_seq_len=512, column_prompt="prompt", column_target="answer"):
#     """
#     Generator that yields batches of (input_ids, attention_mask, target_ids) from the dataset.
#     Shuffles the dataset every epoch.
#     """
#     dataset = dataset.sample(frac=1).reset_index(drop=True)
#     for i in range(0, len(dataset), batch_size):
#         raw_batch = dataset[i:i+batch_size]
#         encoded = encode_batch(raw_batch, tokenizer, max_seq_len, column_prompt, column_target)
#         if encoded is not None:
#             yield encoded

def data_generator(df, tokenizer, batch_size, max_seq_len, 
                   column_prompt="prompt", column_target="answer"):
    """
    Example generator that:
      1) Splits df into batches,
      2) Tokenizes the 'prompt' and 'answer',
      3) Yields input_ids, attention_mask, and labels.

    Modify this to match how you want your data to be loaded.
    """
    # Shuffle or not, depending on preference
    df = df.sample(frac=1.0).reset_index(drop=True)

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        # Convert all prompts to model input
        inputs = tokenizer(
            list(batch_df[column_prompt].values),
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        
        # Convert all targets to labels
        labels = tokenizer(
            list(batch_df[column_target].values),
            padding="longest",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )

        input_batch = inputs["input_ids"].cuda()
        attn_batch = inputs["attention_mask"].cuda()

        # For seq2seq models, the labels usually correspond to "labels"
        # Some models also need a <pad_token_id> mask replaced with -100
        label_batch = labels["input_ids"].cuda()
        label_batch[label_batch == tokenizer.pad_token_id] = -100

        yield input_batch, attn_batch, label_batch