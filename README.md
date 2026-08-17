# Invoice Extraction

Extracts structured data from image-based invoices (scanned PDFs, photographed
invoices, multi-invoice PDF batches) using a vision LLM, with a set of
deterministic checks layered on top to catch hallucination and misattribution
before the data reaches a spreadsheet.

## What it extracts

Per invoice (one row per PDF page): `tax_invoice_no`, `issued_by`, `issued_to`,
`trn`, `address`, `date_of_issue`, `invoice_currency`, `amount_excluding_vat`,
`vat_percent`, `vat_amount`, `total_amount`, `beneficiary_name`,
`beneficiary_bank_name`, `beneficiary_account_number`, `beneficiary_swift_code`.

Each page is also classified (`document_type`, e.g. "Tax Invoice" vs.
"Delivery Note" vs. "Purchase Order") so that supporting documents which
merely *reference* an invoice number don't get extracted as a second,
duplicate invoice row.

## Reliability checks

No generative model can be guaranteed hallucination-free, so this pipeline
layers several deterministic checks on top of the LLM extraction rather than
trusting it outright:

- **OCR cross-verification** — the LLM must return a verbatim snippet for
  every value it extracts; an independent Tesseract OCR pass on the same page
  checks that the snippet is actually present on the page. A value whose
  snippet can't be found is flagged in `unverified_fields`.
- **Non-ASCII identifier check** — invoice numbers, TRNs, account numbers, and
  SWIFT codes should always be plain ASCII; a non-Latin look-alike character
  (e.g. Greek "Τ" instead of Latin "T") is flagged, since this is invisible to
  the eye but breaks exact-match lookups and is easily missed by fuzzy OCR
  matching.
- **VAT arithmetic identity** — flags a mismatch if
  `amount_excluding_vat + vat_amount != total_amount`, and separately flags
  the case where an excl.-VAT figure is present but no VAT rate/amount exists
  anywhere on the invoice (usually means the total was duplicated into the
  wrong field).
- **Document-type enforcement** — `is_invoice: false` pages have every field
  forced to blank in code, regardless of what the model returned, so a
  delivery note or PO can never produce a phantom invoice row.

None of this makes hallucination literally zero — treat `unverified_fields`
and `format_warnings` as a review flag, not a certainty. Known false-positive
source: Tesseract's own OCR errors on a low-quality scan can flag a genuinely
correct value.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract   # OCR verification layer, not extraction itself
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

## Usage

CLI:
```bash
python invoice_extract.py invoices.pdf --out invoices.xlsx
python invoice_extract.py scan1.png scan2.jpg --out invoices.csv
```

Streamlit UI (drag-and-drop upload, live per-page progress, on-screen table,
CSV/Excel download, flagged rows highlighted):
```bash
streamlit run invoice_app.py
```

## Model

Defaults to `openrouter/google/gemini-2.5-flash` via LiteLLM (set
`INVOICE_LLM_MODEL` to use a different vision-capable model). Uses
`OPENROUTER_API_KEY` — any LiteLLM-supported provider works if you change the
model string and corresponding API key env var.
