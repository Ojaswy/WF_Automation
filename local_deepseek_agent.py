import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.pdf_table import PDFTableReader
from llama_index.readers.file import PyMuPDFReader


llm = Ollama(
    model="deepseek-r1:8b", 
    request_timeout=600.0,
    additional_kwargs={"num_ctx": 16384, "temperature": 0} 
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
#Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

# 2. DATA LOADING: Specialized Table Reader
# reader = PDFTableReader()
# input_dir = "data/"
# all_docs = []

# for file in os.listdir(input_dir):
#     if file.endswith(".pdf"):
#         path = os.path.join(input_dir, file)
#         # load_data returns a list of documents, one per table found
#         all_docs.extend(reader.load_data(file=path))


file_extractor = {".pdf": PyMuPDFReader()}
reader = SimpleDirectoryReader(input_dir="data/", file_extractor=file_extractor)
documents = reader.load_data()


# 3. INDEXING
#index = VectorStoreIndex.from_documents(all_docs)
index = VectorStoreIndex.from_documents(documents)


# 4. RECONCILIATION AGENT
query_engine = index.as_query_engine(
    similarity_top_k=11,
    streaming=False
)

reconciliation_prompt = """
You are a local Financial Audit Agent. All documents are already in your memory.
DO NOT ask to upload files. Use the provided context to perform a reconciliation.

RECONCILIATION STEPS:
1. Extract every invoice: [Vendor Name, Invoice #, Grand Total].
2. Extract every bank transaction: [Date, Vendor, Amount].
3. Compare them based on Vendor name and matching amounts.

OUTPUT FORMAT (Markdown Table):
| Category | Vendor | Invoice # | Total | Bank Amount | Status |
|----------|--------|-----------|-------|-------------|--------|

CLASSIFICATION RULES (Use numerical reasoning and logic to recheck and verify the categories):
- EXACT_MATCH: Invoice Total == Bank Amount
- OVERPAID: Bank Amount > Invoice Total
- UNDERPAID: Bank Amount < Invoice Total
- DUPLICATE: A single Invoice exists but the payment is done multiple times according to the bank statement
- MISSED_PAYMENT: Invoice exists, no bank record
- NO_INVOICE_FOUND: Bank record exists, no invoice

Finally, provide a 'Math Verification' section where you show your reasoning behind the classification.
"""

response = query_engine.query(reconciliation_prompt)
print(f"\n--- AUDIT REPORT ---\n{response}")
