import os
import gc
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.encoding import data_generator  # If you use a custom generator
from utils.loss_mT5 import compute_loss_mT5  # If you have a separate script

def training_loop(
    model,
    train_df,
    val_df,
    tokenizer,
    optimizer,
    scheduler,
    start_epoch,
    end_epoch,
    batch_size,
    checkpoint_dir,
    checkpoint_prefix,
    max_seq_len=512,
    use_amp=True,
    accumulation_steps=1,
    column_prompt="prompt",
    column_target="answer",
    print_freq=100,
    fraction_to_use = 1.0
):
    """
    Training loop with TQDM progress bar per epoch, integrated logging (loss, LR),
    optional AMP, gradient accumulation, and checkpointing.
    
    [DEBUG] prints:
      - Shapes of batches
      - Token IDs for the first example in each batch
      - Loss values each step
    """
    print("[INFO] Starting training loop...")
    print(f"[INFO] use_amp={use_amp}, accumulation_steps={accumulation_steps}")
    
    # Automatic Mixed Precision scaler
    scaler = GradScaler(enabled=use_amp)

    # Make sure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Switch model to train mode
    model.train()

    global_step = 0
    losses_overall = []

    for epoch_idx in range(start_epoch, end_epoch + 1):
        max_input_length = min(max_seq_len, start_epoch * 32)
        print(f"\n--- Training epoch {epoch_idx} ---")

        df_shuffled = train_df.sample(frac=fraction_to_use).reset_index(drop=True)
        print(f"[INFO] Fraction of data used: {fraction_to_use:.2f} i.e. {len(df_shuffled)} samples out of {len(train_df)}")
        batch_losses = []
        # Calculate total steps per epoch
        n_batches = int(np.ceil(len(df_shuffled) / batch_size))
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch_idx}", dynamic_ncols=True)

        for batch_idx in pbar:
            start_i = batch_idx * batch_size
            end_i = start_i + batch_size
            batch_df = df_shuffled.iloc[start_i:end_i]

            inputs = tokenizer(
                list(batch_df[column_prompt].values),
                padding="longest", # "max_length",
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            )
            
            labels = tokenizer(
                list(batch_df[column_target].values),
                padding="longest", #"max_length",
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt"
            )

            input_batch = inputs["input_ids"].cuda()
            attn_batch = inputs["attention_mask"].cuda()
            label_batch = labels["input_ids"].cuda()

            label_batch[label_batch == tokenizer.pad_token_id] = -100

            with autocast(enabled=use_amp, device_type="cuda"):
                outputs = model(
                    input_ids=input_batch,
                    attention_mask=attn_batch,
                    labels=label_batch
                )
                loss = outputs.loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            raw_loss_val = loss.item() * accumulation_steps 
            batch_losses.append(raw_loss_val)
            losses_overall.append(raw_loss_val)
            global_step += 1

            if (batch_idx + 1) % print_freq == 0:
                avg_loss = np.mean(batch_losses[-print_freq:])
                lr_current = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr_current:.6f}"})

        val_loss = compute_validation_loss(
            model, val_df, tokenizer, batch_size=batch_size, max_seq_len=max_input_length,
            column_prompt=column_prompt, column_target=column_target, use_amp=use_amp
        )
        if epoch_idx % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}-epoch-{epoch_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"{checkpoint_prefix}-epoch-{epoch_idx}"))

        # Also save the last checkpoint as "last_epoch"
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{checkpoint_prefix}-last_epoch.pt"))
        tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"{checkpoint_prefix}-last_epoch"))

        gc.collect()
        torch.cuda.empty_cache()

    print("[INFO] Training completed.")
    return losses_overall


def compute_validation_loss(
    model,
    val_df,
    tokenizer,
    batch_size=8,
    max_seq_len=512,
    column_prompt="prompt",
    column_target="answer",
    use_amp=True,
    max_iters=100
):
    """
    Simple function to compute average val loss on up to `max_iters` batches.
    If you want the full validation set, remove the break condition.

    [DEBUG] prints:
      - Example shapes and tokens for the first few batches
      - The partial losses
    """
    model.eval()
    losses = []
    scaler = GradScaler(enabled=use_amp)

    # No grads during validation
    with torch.no_grad(), autocast(enabled=use_amp, device_type="cuda"):
        n_batches_val = int(np.ceil(len(val_df) / batch_size))
        # print(f"[INFO] Validation: n_batches_val={n_batches_val}, evaluating up to {max_iters} batches.")
        
        for i in range(n_batches_val):
            if i >= max_iters:
                break

            start_i = i * batch_size
            end_i = start_i + batch_size
            batch_df = val_df.iloc[start_i:end_i]

            inputs = tokenizer(
                list(batch_df[column_prompt].values),
                padding="longest",
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )
            labels = tokenizer(
                list(batch_df[column_target].values),
                padding="longest",
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )

            input_batch = inputs["input_ids"].cuda()
            attn_batch = inputs["attention_mask"].cuda()
            label_batch = labels["input_ids"].cuda()
            label_batch[label_batch == tokenizer.pad_token_id] = -100

            # if i < 2:  # Print a bit of debug info for the first 2 batches
                # print(f"[DEBUG][Val] Batch {i}, input_batch.shape={input_batch.shape}")
                # print(f"           Example input tokens (first 10): {input_batch[0, :10].tolist()}")
                # print(f"           Example label tokens (first 10): {label_batch[0, :10].tolist()}")

            outputs = model(
                input_ids=input_batch,
                attention_mask=attn_batch,
                labels=label_batch
            )
            loss_val = outputs.loss.item()
            losses.append(loss_val)
            
            # if np.isnan(loss_val):
                # print(f"[ERROR] NaN in validation loss at batch {i}!")
                # Potentially break or raise an error

    model.train()
    avg_val_loss = float(np.mean(losses))
    # print(f"[INFO] Validation average loss (over {min(max_iters, n_batches_val)} batches): {avg_val_loss:.4f}")
    return avg_val_loss


def plot_training(losses):
    """
    Plot the smoothed training losses over steps. 
    Adjust window_size if you want more/less smoothing.
    """
    window_size = 50
    if len(losses) < window_size:
        window_size = len(losses)
    smoothed_losses = [
        np.mean(losses[i : i + window_size]) 
        for i in range(len(losses) - window_size)
    ]

    plt.figure()
    plt.plot(smoothed_losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss (Smoothed)")
    plt.show()