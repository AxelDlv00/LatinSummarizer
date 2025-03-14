import pandas as pd

# Function to generate prompts and answers based on various options
def generate_prompt_with_stanza(df, language1, language2, 
                                column_tokenized_1, column_pos_1,
                                column_tokenized_2, column_pos_2,
                                prompt_stanza, answer_stanza, clue):
    """
    Generates a DataFrame with 'prompt' and 'answer' columns according to various tag rules.

    Parameters:
      df               : DataFrame with data.
      language1        : Source language code (e.g. "en").
      language2        : Target language code (e.g. "la").
      column_tokenized_1: Column name for language1 tokenized text.
      column_pos_1     : Column name for language1 POS tags.
      column_tokenized_2: Column name for language2 tokenized text.
      column_pos_2     : Column name for language2 POS tags.
      prompt_stanza    : Boolean. If True, prompt uses "<with_stanza>", else "<no_stanza>".
      answer_stanza    : Boolean. If True, answer uses "<with_stanza>", else "<no_stanza>".
      clue             : Boolean. If True, answer includes a <clue> wrapped section.
    """
    prompt_stanza_tag = "<with_stanza>" if prompt_stanza else "<no_stanza>"
    answer_stanza_tag = "<with_stanza>" if answer_stanza else "<no_stanza>"

    # Function to create the prompt string
    def make_prompt(row):
        pos_answer = " ".join([f"<{pos}>" for pos in row[column_pos_2]])
        
        if prompt_stanza:
            # Alternate between tokenized words and POS tags
            text_1 = " ".join([f"{tok} <{pos}>" for tok, pos in zip(row[column_tokenized_1], row[column_pos_1])])
        else: 
            text_1 = row[language1]  # Use raw text if not using stanza
        
        if clue:
            prompt = (f"<{language1}> {prompt_stanza_tag} {text_1} <{language1}> "
                      f"<clue> {pos_answer} <clue> <{language1}.{language2}> <{language2}> {answer_stanza_tag} ")
        else:
            prompt = f"<{language1}> {prompt_stanza_tag} {text_1} <{language1}> <{language1}.{language2}> <{language2}> {answer_stanza_tag} "
        
        return prompt

    # Function to create the answer string
    def make_answer(row):
        if answer_stanza:
            # Alternate between tokenized words and POS tags
            text_2 = " ".join([f"{tok} <{pos}>" for tok, pos in zip(row[column_tokenized_2], row[column_pos_2])])
        else: 
            text_2 = row[language2]  # Use raw text if not using stanza
        
        answer = f"{text_2} <{language2}>"
        return answer

    # Apply functions to create new DataFrame with prompt and answer
    df_prompt = pd.DataFrame()
    df_prompt["prompt"] = df.apply(make_prompt, axis=1)
    df_prompt["answer"] = df.apply(make_answer, axis=1)
    
    return df_prompt

def generate_prompt_no_stanza(df, language1, language2):
    """ 
    df should have a column 'language1' and 'language2' 
    return :
    df_prompt : DataFrame with columns 'prompt' and 'answer'
    """
    df_prompt = pd.DataFrame()
    df_prompt["prompt"] = df[language1].apply(lambda x: f"<{language1}> {x} <{language1}> <{language1}.{language2}> <{language2}>")
    df_prompt['answer'] = df[language2].apply(lambda x: f"{x} <{language2}>")
    return df_prompt

def generate_prompt_summaries(df, language, input_column, summary_column):
    """ 
    df should have a column 'language1' and 'language2' 
    return :
    df_prompt : DataFrame with columns 'prompt' and 'answer'
    """
    df_prompt = pd.DataFrame()
    df_prompt["prompt"] = df[input_column].apply(lambda x: f"<{language}> {x} <{language}> <{language}.{language}> <{language}>")
    df_prompt['answer'] = df[summary_column].apply(lambda x: f"{x} <{language}>")
    return df_prompt

def generate_prompts_with_distribution(df, language1, language2, column_tokenized_1, column_pos_1,
                                       column_tokenized_2, column_pos_2, distribution):
    """
    Splits the dataframe into several parts based on given distributions and applies 
    'generate_prompt_with_stanza' accordingly.

    Parameters:
      df               : DataFrame with data.
      language1        : Source language code.
      language2        : Target language code.
      column_tokenized_1: Column name for language1 tokenized text.
      column_pos_1     : Column name for language1 POS tags.
      column_tokenized_2: Column name for language2 tokenized text.
      column_pos_2     : Column name for language2 POS tags.
      distribution     : Dictionary mapping (prompt_stanza, answer_stanza, clue) combinations to probabilities.

    Returns:
      A single concatenated DataFrame with all generated prompts.
    """
    dfs = []
    total_samples = len(df)
    
    for (prompt_stanza, answer_stanza, clue), percentage in distribution.items():
        print(f"Generating prompts with distribution: prompt_stanza={prompt_stanza}, answer_stanza={answer_stanza}, clue={clue}, percentage={percentage}")
        sample_size = int(total_samples * percentage)
        print(f"Sample size: {sample_size}/{total_samples}")
        print(f'Lenghts before sampling: {len(df)}')
        sampled_df = df.sample(n=sample_size, replace=False, random_state=42) # replace=False to avoid duplication, so that we can sample without replacement
        df = df.drop(sampled_df.index)  # Remove sampled rows to avoid duplication
        print(f'Lenghts after sampling: {len(df)}, {len(sampled_df)}')
        
        generated_df = generate_prompt_with_stanza(sampled_df, language1, language2, 
                                                   column_tokenized_1, column_pos_1,
                                                   column_tokenized_2, column_pos_2,
                                                   prompt_stanza, answer_stanza, clue)
        dfs.append(generated_df)
    
    return pd.concat(dfs, ignore_index=True)
