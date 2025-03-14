import numpy as np
import torch
from utils.encoding import data_generator

def compute_loss_mT5(model, val_df, tokenizer, 
                     batch_size=32, max_seq_len=512, max_iters=100, 
                     column_prompt="prompt", column_target="answer"):
    """
    Computes the average validation loss over at most `max_iters` minibatches.
    If you want to evaluate on the *entire* validation set, remove the `max_iters` check.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        val_gen = data_generator(
            val_df, tokenizer, batch_size, max_seq_len, 
            column_prompt=column_prompt, column_target=column_target
        )
        for i, batch_data in enumerate(val_gen):
            if i >= max_iters: 
                break  # remove or raise if you want full val pass

            input_batch, attn_batch, label_batch = batch_data
            outputs = model(
                input_ids=input_batch,
                attention_mask=attn_batch,
                labels=label_batch
            )
            losses.append(outputs.loss.item())

    model.train()
    return np.mean(losses)