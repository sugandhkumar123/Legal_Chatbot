import pandas as pd
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# --- 1. Configuration ---
#XLSX_FILE_PATH = "data/web_scrap_vidhikarya_data.xlsx"
CSV_FILE_PATH = "data/enhanced_vidhikarya_data.csv"

COLUMN_TO_JOIN_1 = "question"
COLUMN_TO_JOIN_2 = "updated_answers"
COLUMN_TO_JOIN_3="date_of_question"
FAISS_INDEX_PATH = "vectorstore_filter/db_faiss"


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
    print(f"Selected columns: {df.columns.tolist()}")
    return df

def create_langchain_documents(df, join_col1, join_col2):
    """Converts a pandas DataFrame into a list of LangChain Document objects."""
    print("Creating LangChain documents...")
    documents = []
    metadata_columns = [col for col in df.columns if col not in [join_col1, join_col2,'answers','updated_no_of_answers','Number of Ans']]
    
    for index, row in df.iterrows():
        #page_content = f"Question: {row[join_col1]}\n\nAnswer: {row[join_col2]}"
        page_content = row[join_col2]
        metadata = {col: row[col] for col in metadata_columns}
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Successfully created {len(documents)} documents.")
    return documents

def create_faiss_index(documents, index_path):
    """Creates and saves a FAISS vector store from a list of documents."""
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"Saving FAISS index to {index_path}...")
    vector_store.save_local(index_path)
    print("FAISS index created and saved successfully!")


#

# --- 4. Main Execution ---

if __name__ == "__main__":
    print("Create the FAISS vector index (run this once).")
    print("\n--- Starting Index Creation ---")
    dataframe = load_data(CSV_FILE_PATH)
    print(len(dataframe))
    #print(dataframe)
    conditions = ['2 years ago', "3 years ago", "4 years ago", "5 years ago", "6 years ago", "7 years ago"]
    filtered_dataframe = dataframe[dataframe[COLUMN_TO_JOIN_3].isin(conditions)].copy()
    print(len(filtered_dataframe))
    print(f"Number of rows after filtering for '{', '.join(conditions)}': {len(filtered_dataframe)}")
    #print(filtered_dataframe)
    
    docs = create_langchain_documents(filtered_dataframe, COLUMN_TO_JOIN_1, COLUMN_TO_JOIN_2)
    print(docs[1])
    create_faiss_index(docs, FAISS_INDEX_PATH)
