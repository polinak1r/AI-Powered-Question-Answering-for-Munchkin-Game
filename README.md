## Notebook Outline: AI-Powered Question Answering for Munchkin Game

1. **Imports, Model Setup, and Prompt Functions**  
   We begin by installing and importing the necessary libraries and loading a causal language model (`microsoft/Phi-3-mini-128k-instruct`) along with its tokenizer. We define a `prompt_maker` function to construct the question-and-options prompt and a `score_options` function to compute logits for each option.

2. **Kaggle Setup, Context Truncation, and Corpus Assembly**  
   We configure Kaggle credentials to download the rules (`munchkin_rules.md`) and other files. A `truncate_tokens` function enforces `MAX_LEN = 256` to limit the context size. We then read and sentence-split the Munchkin rules, scrape and clean the official FAQ, and combine both sets of texts into a single corpus for retrieval.

3. **TF-IDF Retrieval Construction**  
   We build a TF-IDF index by tokenizing all corpus documents and computing term frequencies. A sparse TF matrix is multiplied by the IDF values, enabling quick lookups of the most relevant texts for a given query.

4. **Answering Questions with Context**  
   We load the training dataset (multiple-choice Q&A), retrieve the top-k TF-IDF matches for each question, truncate the context if needed, and pass everything to our language model via a prompt. We select the best-scoring answer based on the final token’s logits.

5. **Evaluation and Submission**  
   We compute accuracy on the training set and then apply the same pipeline to the test set, which does not have labels. Finally, we save the predicted answers (one per question) as a CSV file for submission.
