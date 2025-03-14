import sacrebleu
from utils.generate_translation import generate_translation

def calculate_bleu_and_chrf(df, language1, language2, tokenizer, model, max_examples_to_test=500, column_prompt="prompt", column_target="answer", column_generated='generated_text', column_prefix="prefix", max_input_len=512, max_output_len=512):

    # Shuffle the test set
    df_test = df.sample(frac=1).reset_index(drop=True)

    # Extract truth and source texts
    truth = df_test.loc[df_test[column_prefix] == f"{language1}.{language2}", column_prompt].tolist()[:max_examples_to_test]
    src = df_test.loc[df_test[column_prefix] == f"{language1}.{language2}", column_target].tolist()[:max_examples_to_test]
    preds = df_test.loc[df_test[column_prefix] == f"{language1}.{language2}", column_generated].tolist()[:max_examples_to_test]

    # Calculate BLEU and CHRF scores with special tokens
    bleu = sacrebleu.corpus_bleu(preds, [truth])
    chrf = sacrebleu.corpus_chrf(preds, [truth])

    bleu_score = bleu.score
    chrf_score = chrf.score

    print(f"{language1} → {language2} BLEU Score: {bleu_score:.2f}")
    print(f"{language1} → {language2} CHRF Score: {chrf_score:.2f}")

    return bleu_score, chrf_score