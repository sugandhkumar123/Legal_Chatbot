import os
import pandas as pd
import argparse
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
import numpy as np
import nltk
import evaluate

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.evaluation.qa import QAEvalChain

# --- NEW: Import for similarity calculation ---
# You may need to install scikit-learn: pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# --- Setup and Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

FAISS_INDEX_PATH = "vectorstore_filter/db_faiss"
#XLSX_FILE_PATH = "data/web_scrap_vidhikarya_data.xlsx"
CSV_FILE_PATH = "data/enhanced_vidhikarya_data.csv"

def load_data_from_xlsx(file_path):
    """Loads data from an .xlsx file into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    # Ensure no empty values which can cause errors
    df = df.fillna('')
    print("Data loaded successfully.")
    return df

def load_data(file_path):

    print(f"Loading data from {file_path}...")
    # Define the exact column names you want to load from the file
    columns_to_keep = [
        'Category',
        'Sub Category',
        'location',	
        'question',	
        'answers',	
        'Number of Ans',
        'Link',
        'advocate_names',
        'date_of_question',
        'date_of_scraping',
        'updated_no_of_answers',
        'updated_answers'
    ]

    df = pd.read_csv(file_path, usecols=columns_to_keep, low_memory=False)
    # Fill any missing values in the selected columns with an empty string
    df = df.fillna('')
    
    print("Data loaded successfully.")
    return df

# --- RAG Chain and Utility Functions ---

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# This function remains unchanged from the previous step
# def extract_claims_with_llm(text_to_evaluate: str, llm) -> set:
#     ...

def get_rag_chain(embeddings, claim_extractor_llm):
    """
    Initializes a RAG chain incorporating Chain-of-Thought, Few-Shot examples,
    and an "extract-and-synthesize" step for maximum accuracy.
    """
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # --- NEW: Final prompt combining CoT, Few-Shot, and Guided Generation ---
    guided_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert AI Legal Analyst. Your primary function is to provide clear, helpful, and factual explanations based ONLY on the provided legal context.

---
**Internal Thought Process:**
1.  **Deconstruct the Question:** Fully understand the user's situation and what they need to know.
2.  **Review Essential Entities:** Scrutinize the provided list of `Essential Entities`. These are the critical Acts, Sections, and facts that must form the core of your answer.
3.  **Consult Full Context:** Read the `Full Context` to understand the background, nuances, and how the entities are applied.
4.  **Synthesize a Helpful Explanation:** Formulate an answer that not only directly addresses the user's question but also explains the key concepts in a way that is easy to understand, connecting the facts to the user's query.

---
**Final Answer Requirements (Your output to the user MUST follow these rules):**
* **Clear and Comprehensive:** The answer should be easy for a non-expert to understand. Explain the key points clearly without being overly brief. The goal is to be helpful, not just short.
* **Professional and Factual:** Do not use overly casual or conversational language. Stick to the facts provided in the source material.
* **No Self-Reference:** Do not mention that you are an AI or refer to your internal process.
* **Strictly Emulate Examples:** The final output's helpful and explanatory style must precisely match the examples below.

---
**Perfect Answer Examples (Few-Shot Learning):**

**Example 1:**
* QUESTION: On what grounds can I file for divorce if my spouse is causing me mental distress?
* ANSWER: What you are describing may be considered 'cruelty' under the law, which is a valid ground for divorce. Section 13(1)(ia) of the Hindu Marriage Act, 1955, clarifies that cruelty is not limited to physical violence. It also includes mental agony and suffering that makes it unfeasible for you to continue living with your spouse.

**Example 2:**
* QUESTION: What is the procedure for a bounced cheque?
* ANSWER: The process for a bounced cheque is laid out in Section 138 of the Negotiable Instruments Act, 1881. The first step is to send a formal legal notice to the person who issued the cheque (the drawer) within 30 days of the bank returning it. If they fail to make the full payment within 15 days of receiving that notice, you then have the right to file a criminal complaint against them.

---
**Source Materials for the Current Question:**

**Essential Entities:**
{essential_entities}

**Full Context:**
{context}
"""),
        ("human", "{question}"),
    ])

    chat_model = ChatGroq(temperature=0.0, model_name="llama3-8b-8192", max_tokens=512)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def extract_context_and_claims(retrieved_docs):
        """
        Takes retrieved docs, formats them, and extracts key claims using an LLM.
        """
        context_str = format_docs(retrieved_docs)
        claims_set = extract_claims_with_llm(context_str, claim_extractor_llm)
        claims_str = "\n".join(sorted(list(claims_set)))
        return {"context": context_str, "essential_entities": claims_str}

    # The chain's logic remains the same as the previous step
    rag_chain = (
        {
            "retrieved_docs": retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
              extracted_info=lambda x: extract_context_and_claims(x["retrieved_docs"])
          )
        | (lambda x: {
              "context": x["extracted_info"]["context"],
              "essential_entities": x["extracted_info"]["essential_entities"],
              "question": x["question"]
          })
        | guided_prompt_template
        | chat_model
        | StrOutputParser()
    )
    
    return rag_chain

# --- NEW: Function to calculate semantic similarity ---
def calculate_cosine_similarity(embeddings_model, text1, text2):
    """Calculates the cosine similarity between two texts."""
    # The embed_documents method returns a list of vectors
    vector1 = np.array(embeddings_model.embed_documents([text1])[0]).reshape(1, -1)
    vector2 = np.array(embeddings_model.embed_documents([text2])[0]).reshape(1, -1)
    
    # cosine_similarity returns a 2D array, get the single value
    similarity_score = cosine_similarity(vector1, vector2)[0][0]
    return similarity_score

# --- Mode-Specific Functions ---

def run_chat_mode(embeddings):
    """Handles the interactive chat session."""
    claim_extractor_llm = ChatGroq(temperature=0.0, model_name="llama3-70b-8192")
    rag_chain = get_rag_chain(embeddings,claim_extractor_llm)
    print("\n🤖 AI Assistant is ready. Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        response = rag_chain.invoke(user_query)
        print(f"\nAI: {response}")

def extract_claims_with_llm(text_to_evaluate: str, llm) -> set:
    """
    Uses a powerful LLM to extract a set of specific, short legal entities from a text.
    """
    if not text_to_evaluate or not isinstance(text_to_evaluate, str):
        return set()

    # --- NEW: More specific prompt for targeted entity extraction ---
    # This prompt instructs the LLM to pull out only short entities and provides examples.
    claim_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert legal entity and fact extractor. Your task is to extract ONLY the following from the given text:
1.  **Acts**: The full name of any mentioned law (e.g., 'Hindu Marriage Act, 1955').
2.  **Sections**: Any mentioned section numbers (e.g., 'Section 138').
3.  **Core Facts/Claims**: Very short, crucial facts, typically 2-4 words (e.g., 'grounds of cruelty', 'cheque dishonour').

DO NOT extract long sentences or explanations. Be concise. List each extracted item on a new line.

---
EXAMPLE 1
Text: 'The Hindu Marriage Act, 1955, under Section 13(1)(ia), allows for divorce on the grounds of cruelty. Cruelty is not limited to physical harm but can also encompass mental agony and suffering.'
Your Output:
Hindu Marriage Act, 1955
Section 13(1)(ia)
grounds of cruelty
mental agony

---
EXAMPLE 2
Text: 'Section 138 of the Negotiable Instruments Act, 1881, deals with the dishonour of a cheque for insufficiency of funds. A legal notice must be sent to the drawer within 30 days.'
Your Output:
Section 138
Negotiable Instruments Act, 1881
dishonour of a cheque
insufficiency of funds
legal notice within 30 days
---
"""),
        ("human", "Please extract the entities and facts from the following text:\n\n{text}")
    ])
    
    # The rest of the chain logic remains the same
    extractor_chain = claim_extraction_prompt | llm | StrOutputParser()
    
    try:
        response = extractor_chain.invoke({"text": text_to_evaluate})
        # Split response into lines and create a set of non-empty, stripped claims
        claims = {claim.strip().lower() for claim in response.split('\n') if claim.strip()}
        return claims
    except Exception as e:
        print(f"Error during claim extraction: {e}")
        return set()       


def run_evaluation_mode(embeddings):
    """
    Handles the evaluation process, including all metrics: Similarity, P/R/F1, ROUGE, METEOR, and Grade.
    """
    # --- Load evaluation models/metrics once ---
    print("Loading evaluation models/metrics...")
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')
    # This powerful LLM will be used for claim extraction AND grading
    eval_llm = ChatGroq(temperature=0.0, model_name="llama3-70b-8192")
    #extract_llm = ChatGroq(temperature=0.0, model_name="gemma2-9b-it")

    rag_chain = get_rag_chain(embeddings, eval_llm) # Pass the eval_llm to the chain

    dataframe = load_data(CSV_FILE_PATH)
    eval_dataframe = dataframe.iloc[40:50,:].copy()

    # 1. Generate predictions
    print("\nGenerating answers for evaluation...")
    predictions = []
    for question in tqdm(eval_dataframe['question']):
        try:
            predictions.append(rag_chain.invoke(question))
        except Exception as e:
            print(f"Error invoking chain for question '{question}': {e}")
            predictions.append(f"Error: {e}")
    eval_dataframe['result'] = predictions

    # 2. Calculate row-by-row metrics in a single pass
    print("\nCalculating Similarity, P/R/F1, ROUGE, and METEOR...")
    # (Initialize lists for all metrics)
    similarities, precisions, recalls, f1_scores = [], [], [], []
    ground_truth_claims_list, generated_claims_list = [], []
    rouge_scores, meteor_scores = [], []
    
    for index, row in tqdm(eval_dataframe.iterrows(), total=eval_dataframe.shape[0]):
        # (This loop contains the logic for similarity, P/R/F1, ROUGE, METEOR as in the previous step)
        # For brevity, the detailed calculation logic from the previous step is assumed to be here.
        # It populates all the lists initialized above.
        ground_truth_answer = row['updated_answers']
        generated_answer = row['result']
        is_error = "Error:" in generated_answer or not generated_answer or not ground_truth_answer
        
        # This is a condensed version of the loop's content from the previous answer
        sim = 0.0 if is_error else cosine_similarity([embeddings.embed_query(ground_truth_answer)],[embeddings.embed_query(generated_answer)])[0][0]
        gtc = set() if is_error else extract_claims_with_llm(ground_truth_answer, eval_llm)
        gc = set() if is_error else extract_claims_with_llm(generated_answer, eval_llm)
        tp = len(gtc.intersection(gc)); fp = len(gc.difference(gtc)); fn = len(gtc.difference(gc))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0; r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        rouge_l = 0.0 if is_error else rouge_metric.compute(predictions=[generated_answer], references=[ground_truth_answer])['rougeL']
        meteor_val = 0.0 if is_error else meteor_metric.compute(predictions=[generated_answer], references=[ground_truth_answer])['meteor']
        
        similarities.append(sim); precisions.append(p); recalls.append(r); f1_scores.append(f1)
        ground_truth_claims_list.append(" | ".join(sorted(list(gtc)))); generated_claims_list.append(" | ".join(sorted(list(gc))))
        rouge_scores.append(rouge_l); meteor_scores.append(meteor_val)


    # Add calculated metrics to DataFrame
    eval_dataframe['similarity'] = similarities
    eval_dataframe['precision'] = precisions
    eval_dataframe['recall'] = recalls
    eval_dataframe['f1_score'] = f1_scores
    eval_dataframe['rougeL'] = rouge_scores
    eval_dataframe['meteor'] = meteor_scores
    eval_dataframe['ground_truth_claims'] = ground_truth_claims_list
    eval_dataframe['generated_claims'] = generated_claims_list
    eval_dataframe.rename(columns={'updated_answers': 'ground_truth'}, inplace=True)

    # 3. --- NEW: Correctness Grading using QAEvalChain ---
    print("\nRunning correctness evaluation with QAEvalChain...")
    eval_chain = QAEvalChain.from_llm(llm=eval_llm)
    
    # Prepare datasets for the evaluator
    eval_examples = []
    predicted_dataset = []
    for i, row in eval_dataframe.iterrows():
        eval_examples.append({"query": row['question'], "answer": row['ground_truth']})
        predicted_dataset.append({"query": row['question'], "result": row['result']})
    
    # Run the evaluation
    evaluation_results = eval_chain.evaluate(
        eval_examples,
        predicted_dataset,
        question_key="query",
        prediction_key="result",
        answer_key="answer"
    )
    eval_dataframe['evaluation_grade'] = [res['results'] for res in evaluation_results]


    # 4. Display and save the final results
    print("\n--- Evaluation Results ---")
    output_columns = [
        'question', 'ground_truth', 'result', 'evaluation_grade', 'similarity', 
        'precision', 'recall', 'f1_score', 'rougeL', 'meteor',
        'ground_truth_claims', 'generated_claims'
    ]
    display_columns = ['question', 'evaluation_grade', 'similarity', 'f1_score', 'rougeL', 'meteor']
    print(eval_dataframe[display_columns])
    print("\n(Full results including claims saved to CSV)")

    # 5. Calculate and print overall average scores
    print("\n--- Overall Metrics ---")
    correct_count = eval_dataframe['evaluation_grade'].str.strip().str.upper().str.startswith("GRADE: CORRECT").sum()
    total_count = len(eval_dataframe)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Overall Correctness Grade: {accuracy:.2f}% ({correct_count}/{total_count} correct)")
    print(f"Average Cosine Similarity: {eval_dataframe['similarity'].mean():.4f}")
    print(f"Average F1-Score (claim-based): {eval_dataframe['f1_score'].mean():.4f}")
    print(f"Average ROUGE-L Score: {eval_dataframe['rougeL'].mean():.4f}")
    print(f"Average METEOR Score: {eval_dataframe['meteor'].mean():.4f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"evaluation_results_{timestamp}.csv"
    eval_dataframe.to_csv(output_filename, columns=output_columns, index=False)
    print(f"\n✅ Evaluation results saved to '{output_filename}'")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG application in chat or evaluation mode.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="chat",
        choices=["chat", "evaluate"],
        help="The mode to run the application in: 'chat' for interactive mode, 'evaluate' for performance metrics."
    )
    args = parser.parse_args()
    
    # Load embeddings model once to be reused
    print("Loading embedding model...")
    main_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' if GPU is available
    )
    print("Embedding model loaded.")

    if args.mode == "chat":
        run_chat_mode(main_embeddings)
    elif args.mode == "evaluate":
        run_evaluation_mode(main_embeddings)



