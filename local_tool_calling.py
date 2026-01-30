import json
from collections import defaultdict
from typing import Dict, List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


# MODEL SETUP

llm = Ollama(
    model="deepseek-r1:8b",
    request_timeout=1800,
    additional_kwargs={"num_ctx": 16384, "temperature": 0},
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# LOAD DOCUMENTS

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


# AGENT PROMPTS

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


# DETERMINISTIC RECONCILIATION TOOL

def reconciliation_tool(extracted_data: Dict) -> Dict:
    invoices = extracted_data.get("invoices", [])
    transactions = extracted_data.get("bank_transactions", [])

    results = {
        "EXACT_MATCH": [],
        "OVERPAID": [],
        "UNDERPAID": [],
        "DUPLICATE_PAYMENT": [],
        "MISSED_PAYMENT": [],
        "NO_INVOICE_FOUND": [],
    }

    invoice_map = {}
    for inv in invoices:
        invoice_map[inv["invoice_number"]] = inv

    payments_by_invoice = defaultdict(list)
    unmatched_transactions = []

    for tx in transactions:
        inv_no = tx.get("invoice_number")
        if inv_no and inv_no in invoice_map:
            payments_by_invoice[inv_no].append(tx)
        else:
            unmatched_transactions.append(tx)

    # Invoice-based reconciliation
    for inv_no, inv in invoice_map.items():
        payments = payments_by_invoice.get(inv_no, [])
        total_paid = sum(p["amount"] for p in payments)

        if not payments:
            results["MISSED_PAYMENT"].append({
                "vendor": inv["vendor"],
                "invoice": inv_no,
                "invoice_total": inv["grand_total"],
            })
            continue

        if len(payments) > 1:
            results["DUPLICATE_PAYMENT"].append({
                "vendor": inv["vendor"],
                "invoice": inv_no,
                "bank_amount": total_paid,
                "count": len(payments),
            })

        if total_paid == inv["grand_total"]:
            results["EXACT_MATCH"].append({
                "vendor": inv["vendor"],
                "invoice": inv_no,
                "invoice_total": inv["grand_total"],
                "bank_amount": total_paid,
            })
        elif total_paid > inv["grand_total"]:
            results["OVERPAID"].append({
                "vendor": inv["vendor"],
                "invoice": inv_no,
                "invoice_total": inv["grand_total"],
                "bank_amount": total_paid,
                "balance_left": total_paid - inv["grand_total"],
            })
        else:
            results["UNDERPAID"].append({
                "vendor": inv["vendor"],
                "invoice": inv_no,
                "invoice_total": inv["grand_total"],
                "bank_amount": total_paid,
                "amount_owed": inv["grand_total"] - total_paid,
            })

    # Unmatched bank transactions
    for tx in unmatched_transactions:
        results["NO_INVOICE_FOUND"].append({
            "date": tx["date"],
            "vendor": tx["vendor"],
            "amount": tx["amount"],
            "description": tx.get("description", ""),
        })

    return results


# AGENT EXECUTION FUNCTIONS

def run_extraction_agent() -> Dict:
    response = query_engine.query(EXTRACTION_AGENT_PROMPT)
    return json.loads(response.response)


def run_reconciliation_tool(extracted_data: Dict) -> Dict:
    # Agentic tool invocation (explicit, deterministic)
    return reconciliation_tool(extracted_data)


def run_manager_agent(reconciliation_results: Dict) -> str:
    prompt = (
        MANAGER_AGENT_PROMPT
        + "\n\nRECONCILIATION JSON:\n"
        + json.dumps(reconciliation_results, indent=2)
    )
    response = query_engine.query(prompt)
    return response.response


# ORCHESTRATION PIPELINE

def run_full_reconciliation_pipeline():
    print("🔍 Running Extraction Agent...")
    extracted_data = run_extraction_agent()

    print("🛠️ Running Deterministic Reconciliation Tool...")
    reconciliation_results = run_reconciliation_tool(extracted_data)

    print("🧠 Running Manager Agent...")
    final_output = run_manager_agent(reconciliation_results)

    print("\n\n✅ FINAL RECONCILIATION REPORT:\n")
    print(final_output)


# RUN PIPELINE
if __name__ == "__main__":
    run_full_reconciliation_pipeline()
