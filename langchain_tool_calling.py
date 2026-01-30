import json
from collections import defaultdict
from typing import Dict, List

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# MODEL SETUP
# ============================================================

llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0,
    num_ctx=16384,
    request_timeout=1800,
)


# ============================================================
# PROMPTS (UNCHANGED SEMANTICS)
# ============================================================

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


# ============================================================
# DETERMINISTIC RECONCILIATION TOOL
# ============================================================

def reconciliation_tool_fn(extracted_data: Dict) -> Dict:
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

    invoice_map = {inv["invoice_number"]: inv for inv in invoices}
    payments_by_invoice = defaultdict(list)
    unmatched_transactions = []

    for tx in transactions:
        inv_no = tx.get("invoice_number")
        if inv_no and inv_no in invoice_map:
            payments_by_invoice[inv_no].append(tx)
        else:
            unmatched_transactions.append(tx)

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

    for tx in unmatched_transactions:
        results["NO_INVOICE_FOUND"].append({
            "date": tx["date"],
            "vendor": tx["vendor"],
            "amount": tx["amount"],
            "description": tx.get("description", ""),
        })

    return results


ReconciliationTool = StructuredTool.from_function(
    name="reconciliation_tool",
    description="Deterministic invoice-to-bank reconciliation tool",
    func=reconciliation_tool_fn,
)


# ============================================================
# AGENT EXECUTION FUNCTIONS
# ============================================================

def run_extraction_agent(context_text: str) -> Dict:
    """
    context_text = combined OCR / PDF text
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=EXTRACTION_AGENT_PROMPT),
        HumanMessage(content=context_text),
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    return json.loads(response)


def run_reconciliation_tool(extracted_data: Dict) -> Dict:
    # Explicit agentic tool invocation (no LLM involved)
    return ReconciliationTool.invoke({"extracted_data": extracted_data})


def run_manager_agent(reconciliation_results: Dict) -> str:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=MANAGER_AGENT_PROMPT),
        HumanMessage(
            content="RECONCILIATION JSON:\n"
            + json.dumps(reconciliation_results, indent=2)
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


# ============================================================
# ORCHESTRATION PIPELINE
# ============================================================

def run_full_reconciliation_pipeline(document_text: str):
    print("🔍 Running Extraction Agent...")
    extracted_data = run_extraction_agent(document_text)

    print("🛠️ Running Deterministic Reconciliation Tool...")
    reconciliation_results = run_reconciliation_tool(extracted_data)

    print("🧠 Running Manager Agent...")
    final_output = run_manager_agent(reconciliation_results)

    print("\n\n✅ FINAL RECONCILIATION REPORT:\n")
    print(final_output)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    loader = PyPDFDirectoryLoader('data2/')
    documents = loader.load()
    combined_text = "\n\n".join(doc.page_content for doc in documents)
    run_full_reconciliation_pipeline(combined_text)
