import pandas as pd
import re
import numpy as np

# Define aggressive regex patterns for filtering
STRICT_PATTERNS = {
    "html_tags": r"<[^>]+>",  # Remove HTML tags like <p>, <a href="...">
    "url_links": r"http[s]?://\S+",  # Remove URLs
    "email_addresses": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Remove emails
    "wikipedia_references": r"\[\s*.*?\s*\]|\(.*?ref.*?\)",  # Remove Wikipedia refs like [1], (ex ref)
    "unwanted_symbols": r"[∂∇∞√±≤≥≠≈×÷∫⇒↑→←⋅°∏∑§©®™µ…·●♠♦♣♥♪♫☀☁★☆☂☃⚡❄]",  # Remove special symbols
    "non_ascii": r"[^\x00-\x7F]+",  # Remove non-ASCII characters
    "digits_only": r"^\d+$",  # Remove rows with only numbers
    "punctuation_only": r"^[^a-zA-Z0-9]+$",  # Remove rows that contain only punctuation
    "junk_brackets": r"[\[\]{}()<>]",  # Remove stray brackets if they exist alone
}

from html import unescape

def clean_text(text, patterns):
    if pd.isna(text) or text.strip() == "":
        return None

    text = unescape(text)  # Convert &lt; to <, &gt; to >, etc.

    for _, pattern in patterns.items():
        text = re.sub(pattern, "", text)

    return text.strip() if text.strip() else None



def clean_up_df(df, columns, verbose=True, min_len=0, max_len=np.inf):
    """
    Cleans a DataFrame by removing unwanted text, symbols, invalid structure, and duplicates.
    
    :param df: Pandas DataFrame containing text columns
    :param columns: List of columns to clean
    :param verbose: Whether to print logs of operations
    :param min_len: Minimum character length for valid text
    :param max_len: Maximum character length for valid text
    :return: Cleaned DataFrame
    """
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if verbose:
        print(f"Initial dataset shape: {df.shape}")
        print()
    
    # Statistics on number of Token for each column before cleaning (mean, std, min, max)
    if verbose:
        for col in columns:
            print('-'*50)
            print(f" - Statistics for column '{col}':")
            # Ensure we only apply split() on valid strings
            tokenized = df[col].dropna().astype(str).apply(lambda x: len(x.split()))
            print(f'    - Mean: {tokenized.mean():.2f}')
            print(f'    - Std: {tokenized.std():.2f}')
            print(f'    - Min: {tokenized.min()}')
            print(f'    - Max: {tokenized.max()}')
            print('-'*50)
        print()


    len_before = len(df)

    # Drop NaN values only from relevant columns
    df = df.dropna(subset=columns)
    if verbose:
        print(f"Removed {len_before - len(df)} rows with NaN values.")
        len_before = len(df)

    # Apply regex cleaning to each specified column
    for col in columns:
        df[col] = df[col].apply(lambda text: clean_text(text, STRICT_PATTERNS))

    # Drop rows where any cleaned column became empty
    df = df.dropna(subset=columns)
    if verbose:
        print(f"Removed {len_before - len(df)} rows with unwanted patterns.")
        len_before = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()
    if verbose:
        print(f"Removed {len_before - len(df)} duplicate rows.")
        len_before = len(df)

    # Remove rows where for this row, all columns have the same value
    df = df.drop_duplicates(subset=columns, keep=False)
    if verbose:
        print(f"Removed {len_before - len(df)} rows with identical values in all columns.")
        len_before = len(df)

    # Remove rows that do not contain Roman letters in any column
    df = df[df.apply(lambda row: any(re.search(r"[A-Za-z]", str(row[col])) for col in columns), axis=1)]
    if verbose:
        print(f"Removed {len_before - len(df)} rows without Roman letters.")
        len_before = len(df)

    # Remove text that is too short or too long
    for col in columns:
        df = df[df[col].str.len().between(min_len, max_len)]
    
    # Statistics on number of tokens for each column before cleaning (mean, std, min, max)
    if verbose:
        for col in columns:
            print('-'*50)
            print(f" - Statistics for column '{col}':")
            
            # Ensure we only apply split() on valid strings
            tokenized = df[col].dropna().astype(str).apply(lambda x: len(x.split()))
            
            print(f'    - Mean: {tokenized.mean():.2f}')
            print(f'    - Std: {tokenized.std():.2f}')
            print(f'    - Min: {tokenized.min()}')
            print(f'    - Max: {tokenized.max()}')
            print('-'*50)
        print()


    return df
