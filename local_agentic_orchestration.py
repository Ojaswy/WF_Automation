import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


#MODEL SETUP 
llm = Ollama(model="deepseek-r1:8b", request_timeout=1800, additional_kwargs={"num_ctx": 16384, "temperature": 0})
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


#LOAD DOCUMENTS

reader = SimpleDirectoryReader(input_dir="data2/")
documents = reader.load_data()

storage_context = StorageContext.from_defaults()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

reranker = FlagEmbeddingReranker(
    top_n=len(documents),
    model="BAAI/bge-reranker-large",
)

query_engine = index.as_query_engine(
    similarity_top_k=len(documents),
    node_postprocessors=[reranker],
    llm=llm,
)


#AGENT PROMPTS

EXTRACTION_AGENT_PROMPT = """
You are an expert financial document extraction agent.

Your job is to extract structured data from:
- Multiple invoice PDFs
- One bank statement PDF

STRICT OUTPUT FORMAT (JSON ONLY — no commentary):

{
  "invoices": [
    {
      "invoice_number": "...",
      "vendor": "...",
      "invoice_date": "...",
      "grand_total": 0.00
    }
  ],
  "bank_transactions": [
    {
      "date": "...",
      "vendor": "...",
      "amount": 0.00,
      "invoice_number": null,
      "description": "..."
    }
  ]
}

Rules:
- Normalize vendor names.
- Amounts must be numeric.
- If invoice number not found in bank transaction, use null.
- No hallucination. If missing, omit.
- No markdown. Output JSON only.
"""


RECONCILIATION_AGENT_PROMPT = """
You are a financial reconciliation agent.

You will receive structured JSON containing:
- invoices
- bank_transactions

Your task:
1. Match invoices to bank transactions using invoice_number first, then vendor + amount.
2. Classify invoices into:
   - EXACT_MATCH
   - OVERPAID
   - UNDERPAID
   - DUPLICATE_PAYMENT
   - MISSED_PAYMENT
3. Classify unmatched bank transactions as:
   - NO_INVOICE_FOUND

STRICT OUTPUT FORMAT (JSON ONLY):

{
  "EXACT_MATCH": [
    {
      "vendor": "...",
      "invoice": "...",
      "invoice_total": 0.00,
      "bank_amount": 0.00
    }
  ],
  "OVERPAID": [
    {
      "vendor": "...",
      "invoice": "...",
      "invoice_total": 0.00,
      "bank_amount": 0.00,
      "balance_left": 0.00
    }
  ],
  "UNDERPAID": [
    {
      "vendor": "...",
      "invoice": "...",
      "invoice_total": 0.00,
      "bank_amount": 0.00,
      "amount_owed": 0.00
    }
  ],
  "DUPLICATE_PAYMENT": [
    {
      "vendor": "...",
      "invoice": "...",
      "bank_amount": 0.00,
      "count": 2
    }
  ],
  "MISSED_PAYMENT": [
    {
      "vendor": "...",
      "invoice": "...",
      "invoice_total": 0.00
    }
  ],
  "NO_INVOICE_FOUND": [
    {
      "date": "...",
      "vendor": "...",
      "amount": 0.00,
      "description": "..."
    }
  ]
}

Rules:
- Partial payments = UNDERPAID
- Overpayments include balance_left
- Duplicate payments require same invoice paid more than once
- No commentary. Output JSON only.
"""


MANAGER_AGENT_PROMPT = """
You are the manager agent responsible for orchestration and final output formatting.

You will:
1. Receive extracted structured data.
2. Pass it to the reconciliation agent.
3. Validate correctness and consistency.
4. Produce final human-readable output.

FINAL OUTPUT FORMAT:

### EXACT_MATCH
- Vendor | Invoice | Invoice Total | Bank Amount

### OVERPAID
- Vendor | Invoice | Invoice Total | Bank Amount | Balance Left

### UNDERPAID
- Vendor | Invoice | Invoice Total | Bank Amount | Amount Owed

### DUPLICATE_PAYMENT
- Vendor | Invoice | Bank Amount | Count

### MISSED_PAYMENT
- Vendor | Invoice | Invoice Total

### NO_INVOICE_FOUND
- Date | Vendor | Amount | Description

Then provide a short human-readable explanation.

Do NOT hallucinate. Use only provided reconciliation results.
"""


#AGENT EXECUTION FUNCTIONS

def run_extraction_agent():
    response = query_engine.query(EXTRACTION_AGENT_PROMPT)
    return json.loads(response.response)


def run_reconciliation_agent(extracted_data):
    prompt = RECONCILIATION_AGENT_PROMPT + "\n\nINPUT JSON:\n" + json.dumps(extracted_data, indent=2)
    response = query_engine.query(prompt)
    return json.loads(response.response)


def run_manager_agent(reconciliation_results):
    prompt = MANAGER_AGENT_PROMPT + "\n\nRECONCILIATION JSON:\n" + json.dumps(reconciliation_results, indent=2)
    response = query_engine.query(prompt)
    return response.response


#ORCHESTRATION PIPELINE

def run_full_reconciliation_pipeline():
    print("🔍 Running Extraction Agent...")
    extracted_data = run_extraction_agent()

    print("📊 Running Reconciliation Agent...")
    reconciliation_results = run_reconciliation_agent(extracted_data)

    print("🧠 Running Manager Agent...")
    final_output = run_manager_agent(reconciliation_results)

    print("\n\n✅ FINAL RECONCILIATION REPORT:\n")
    print(final_output)


#RUN PIPELINE
if __name__ == "__main__":
    run_full_reconciliation_pipeline()
