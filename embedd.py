import pandas as pd
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# --- 1. Configuration ---
XLSX_FILE_PATH = "data/enhanced_vidhikarya_data.xlsx"

COLUMN_TO_JOIN_1 = "question"
COLUMN_TO_JOIN_2 = "answers"
FAISS_INDEX_PATH = "vectorstore/db_faiss"

# --- 2. Helper Functions (for one-time setup) ---

def load_data_from_xlsx(file_path):
    """Loads data from an .xlsx file into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    df = df.fillna('')
    print("Data loaded successfully.")
    return df

def create_langchain_documents(df, join_col1, join_col2):
    """Converts a pandas DataFrame into a list of LangChain Document objects."""
    print("Creating LangChain documents...")
    documents = []
    metadata_columns = [col for col in df.columns if col not in [join_col1, join_col2]]
    
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
    dataframe = load_data_from_xlsx(XLSX_FILE_PATH)
    docs = create_langchain_documents(dataframe, COLUMN_TO_JOIN_1, COLUMN_TO_JOIN_2)
    print(docs[1])
    create_faiss_index(docs, FAISS_INDEX_PATH)