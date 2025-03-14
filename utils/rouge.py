import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm

def calculate_rouge_and_bertscore(df, tokenizer, model, 
                                  max_examples_to_test=300, 
                                  column_prompt="prompt", 
                                  column_target="answer", 
                                  column_generated='generated_text',
                                  column_prefix="prefix", 
                                  language="la",
                                  max_input_len=1000, 
                                  max_output_len=512):
    
    df_test = df.sample(min(len(df), max_examples_to_test)).reset_index(drop=True)  # Sample if dataset is large

    # Extract truth and source texts
    src = df_test.loc[df_test[column_prefix] == f"{language}.{language}", column_prompt].tolist()[:max_examples_to_test]
    truth = df_test.loc[df_test[column_prefix] == f"{language}.{language}", column_target].tolist()[:max_examples_to_test]
    preds = df_test.loc[df_test[column_prefix] == f"{language}.{language}", column_generated].tolist()[:max_examples_to_test]

    # ROUGE scorer
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Compute ROUGE scores
    rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }

    for pred, ref in zip(preds, src):
        scores = rouge.score(pred, ref)
        for key in scores:
            rouge_scores[key].append(scores[key].fmeasure)  # Use F1-score

    avg_rouge = {key: sum(vals) / len(vals) for key, vals in rouge_scores.items()}

    # Compute BERTScore
    _, _, bert_f1 = bert_score(preds, src, lang="la", rescale_with_baseline=True)
    avg_bertscore = torch.mean(bert_f1).item()

    return {
        "ROUGE-1": avg_rouge["rouge1"],
        "ROUGE-2": avg_rouge["rouge2"],
        "ROUGE-L": avg_rouge["rougeL"],
        "BERTScore-F1": avg_bertscore
    }
