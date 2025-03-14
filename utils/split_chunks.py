import pandas as pd
import re

# Function to split text into chunks of approximately 400 words while preserving sentence boundaries
def split_into_chunks(text, chunk_size=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text by sentence boundaries
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        
        if current_length + sentence_length > chunk_size:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_length = sentence_length
        else:
            # Add sentence to the current chunk
            current_chunk.extend(words)
            current_length += sentence_length
    
    # Add any remaining text as a final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks