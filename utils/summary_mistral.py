import torch
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download
import re

###############################################################################
# MODEL LOADING
###############################################################################

def load_model(
    # MODEL_PATH="/Data/AxelDlv/mistral-7b",
    MODEL_PATH="/Data/AxelDlv/Mistral-7B-Instruct-v0.3",
    tokenizer_name="tokenizer.model.v3",
    DOWNLOAD_MODEL=False
):
    """
    Load Mistral model and tokenizer from the specified path.
    If DOWNLOAD_MODEL=True, it will pull them from huggingface_hub first.
    """
    if DOWNLOAD_MODEL:
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            allow_patterns=[
                "params.json",
                "consolidated.safetensors",
                "tokenizer.model.v3"
            ],
            local_dir=MODEL_PATH,
        )
    tokenizer = MistralTokenizer.from_file(f"{MODEL_PATH}/{tokenizer_name}")
    model = Transformer.from_folder(MODEL_PATH)
    return model, tokenizer

def free_model(model, tokenizer):
    """
    Cleanly free GPU memory if needed.
    """
    del model
    del tokenizer
    torch.cuda.empty_cache()

###############################################################################
############################### SUMMARIZATION #################################
###############################################################################

def mistral_summarize_texts(text, model, tokenizer, max_tokens=512, temperature=0.3, instruction="Summarize the following text concisely:"):
    """
    Summarizes a text using the Mistral model, ensuring output is not cut off.
    """
    prompt = f"[INST] {instruction}\n{text} [/INST]"

    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate(
        [tokens], 
        model, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )

    # Ensure proper decoding
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]).strip()
    
    return result