from torch.amp import autocast, GradScaler
import torch
import numpy as np

def generate_translation(input_text, tokenizer, model, src_lang, tgt_lang, max_input_len=512, max_output_len=512, skip_special_tokens=True):
    """
    Generate a translation from `src_lang` to `tgt_lang`.
    src_lang, tgt_lang are strings like 'en', 'la'.
    """
    # Build the model input
    suffix_token = f"<{src_lang}.{tgt_lang}>"
    prompt = f"<{src_lang}> {input_text} <{src_lang}> {suffix_token} <{tgt_lang}>"

    tokenized = tokenizer(
        prompt,
        padding='max_length',  # Pad if necessary
        truncation=True, 
        max_length=max_input_len,
        return_tensors='pt'
    )
    
    input_ids = tokenized["input_ids"].cuda()
    attention_mask = tokenized["attention_mask"].cuda()

    # Generate with `max_new_tokens`
    output_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        max_new_tokens=max_output_len,
    )

    return prompt, tokenizer.decode(output_tokens[0], skip_special_tokens=skip_special_tokens)

def generate_translation_with_options(input_text, tokenizer, model, src_lang, tgt_lang,
                         tokenized_src=None, pos_src=None,
                         tokenized_tgt=None, pos_tgt=None,
                         prompt_stanza=False, answer_stanza=False, clue=False,
                         max_input_len=512, max_output_len=512, skip_special_tokens=True):
    """
    Generate a translation from `src_lang` to `tgt_lang` by constructing a prompt 
    that follows the same tag conventions as your prompt-generation functions.
    
    Parameters:
      input_text     : Raw input text in the source language.
      tokenizer      : The Hugging Face tokenizer.
      model          : The LoRA-adapted translation model.
      src_lang       : Source language code (e.g., 'en').
      tgt_lang       : Target language code (e.g., 'la').
      tokenized_src  : (Optional) List of tokenized words for the source text.
      pos_src        : (Optional) List of POS tags for the source text.
      tokenized_tgt  : (Optional) List of tokenized words for the target text (for clue).
      pos_tgt        : (Optional) List of POS tags for the target text.
      prompt_stanza  : Boolean; if True, include stanza tags in the prompt.
      answer_stanza  : Boolean; if True, include stanza tags in the answer (output) part.
      clue           : Boolean; if True, include a <clue> section constructed from target POS tags.
      max_input_len  : Maximum number of tokens for the input prompt.
      max_output_len : Maximum number of tokens to generate.
      skip_special_tokens : Whether to skip special tokens during decoding.
      
    Returns:
      A string containing the generated translation.
    """
    # Determine stanza tags based on flags
    prompt_stanza_tag = "<with_stanza>" if prompt_stanza else "<no_stanza>"
    answer_stanza_tag = "<with_stanza>" if answer_stanza else "<no_stanza>"

    if src_lang == tgt_lang:
        prompt_stanza_tag = ""
        answer_stanza_tag = ""
    # Construct the main text portion for the prompt.
    # If using stanza, alternate tokens with POS tags; otherwise, use the raw text.
    if prompt_stanza and tokenized_src is not None and pos_src is not None:
        text_1 = " ".join([f"{tok} <{pos}>" for tok, pos in zip(tokenized_src, pos_src)])
    else:
        text_1 = input_text
    
    # Build the prompt.
    # If a clue is desired and target POS info is provided, include it in a <clue> section.
    if clue and tokenized_tgt is not None and pos_tgt is not None:
        pos_answer = " ".join([f"<{pos}>" for pos in pos_tgt])
        prompt = (
            f"<{src_lang}> {prompt_stanza_tag} {text_1} <{src_lang}> "
            f"<clue> {pos_answer} <clue> <{src_lang}.{tgt_lang}> <{tgt_lang}> {answer_stanza_tag}"
        )
    else:
        prompt = f"<{src_lang}> {prompt_stanza_tag} {text_1} <{src_lang}> <{src_lang}.{tgt_lang}> <{tgt_lang}> {answer_stanza_tag}"
    
    # Tokenize the prompt.
    tokenized = tokenizer(
        prompt,
        padding='max_length',
        truncation=True,
        max_length=max_input_len,
        return_tensors='pt'
    )
    
    input_ids = tokenized["input_ids"].cuda()
    attention_mask = tokenized["attention_mask"].cuda()

    # Generate output tokens.
    output_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        max_new_tokens=max_output_len,
    )

    return prompt, tokenizer.decode(output_tokens[0], skip_special_tokens=skip_special_tokens)

def inference_from_csv(
    model,
    test_df,
    tokenizer,
    batch_size=8,
    max_seq_len=512,
    column_prompt="prompt",
    use_amp=True
):
    """
    Perform inference on a test DataFrame that contains prompts.

    Parameters:
      model          : The trained model (e.g. mT5 with LoRA) in evaluation mode.
      test_df        : A pandas DataFrame containing test data.
      tokenizer      : The tokenizer corresponding to the model.
      batch_size     : Number of samples to process at once.
      max_seq_len    : Maximum sequence length for tokenization.
      column_prompt  : Column name in test_df that contains the prompts.
      use_amp        : Boolean flag to enable automatic mixed precision.

    Returns:
      results        : A list of dictionaries, each containing the original prompt and the generated text.
    """
    model.eval()  # Ensure model is in eval mode
    results = []
    n_batches = int(np.ceil(len(test_df) / batch_size))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = test_df.iloc[start_idx:end_idx]
        prompts = list(batch_df[column_prompt].values)
        
        # Tokenize the batch of prompts
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].cuda()
        attention_mask = tokenized["attention_mask"].cuda()
        
        # Generate outputs using no_grad and AMP (if enabled)
        with torch.no_grad(), autocast(enabled=use_amp, device_type="cuda"):
            output_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                max_new_tokens=512,
            )
        
        # Decode generated tokens and collect results
        for prompt, output in zip(prompts, output_tokens):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generated_text": generated_text
            })
    
    return results

def inference_from_csv_adding_column(
    model,
    test_df,
    tokenizer,
    batch_size=8,
    max_seq_len=512,
    column_prompt="prompt",
    new_column="generated_text",
    use_amp=True
):
    """
    Perform inference on a test DataFrame that contains prompts.

    Parameters:
      model          : The trained model (e.g. mT5 with LoRA) in evaluation mode.
      test_df        : A pandas DataFrame containing test data.
      tokenizer      : The tokenizer corresponding to the model.
      batch_size     : Number of samples to process at once.
      max_seq_len    : Maximum sequence length for tokenization.
      column_prompt  : Column name in test_df that contains the prompts.
      new_column     : Column name to store the generated text.
      use_amp        : Boolean flag to enable automatic mixed precision.

    Returns:
        test_df        : The updated DataFrame with the new column added.
    """
    model.eval()  # Ensure model is in eval mode
    results = []
    n_batches = int(np.ceil(len(test_df) / batch_size))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = test_df.iloc[start_idx:end_idx]
        prompts = list(batch_df[column_prompt].values)
        
        # Tokenize the batch of prompts
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].cuda()
        attention_mask = tokenized["attention_mask"].cuda()
        
        # Generate outputs using no_grad and AMP (if enabled)
        with torch.no_grad(), autocast(enabled=use_amp, device_type="cuda"):
            output_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                max_new_tokens=512,
            )
        
        # Decode generated tokens and collect results
        for prompt, output in zip(prompts, output_tokens):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            results.append(generated_text)

    test_df[new_column] = results
    return test_df