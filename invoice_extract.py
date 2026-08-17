"""Extract invoice fields from image-based PDFs/images using a vision LLM.

Each PDF page (or each standalone image file) is treated as one invoice, so a
multi-invoice PDF (e.g. one scanned invoice per page) yields one output row
per page. Requires OPENROUTER_API_KEY in the environment (see .env.example).

Usage:
    python invoice_extract.py invoices.pdf --out invoices.xlsx
    python invoice_extract.py scan1.png scan2.jpg --out invoices.csv
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import litellm
import pandas as pd
import pypdfium2 as pdfium
import pytesseract
from dotenv import load_dotenv
from PIL import Image
from pydantic import ValidationError, create_model

load_dotenv()

DEFAULT_MODEL = os.environ.get("INVOICE_LLM_MODEL", "openrouter/google/gemini-2.5-flash")
RENDER_SCALE = 2.0  # ~144 DPI: legible for OCR-by-vision without huge payloads
FUZZY_MATCH_THRESHOLD = 0.75  # snippet-vs-OCR similarity required to count as "verified"

# Single source of truth for the schema: (field_name, description shown to the LLM).
# Drives the prompt's schema block, the InvoiceRecord model, and the output columns,
# so the three can never drift out of sync with each other.
FIELD_DEFS = [
    ("tax_invoice_no", 'the tax invoice number, exactly as printed (may be labeled "Tax Invoice No.", "Invoice No.", etc.)'),
    ("issued_by", "vendor / seller name"),
    ("issued_to", "customer / buyer name"),
    ("trn", "Tax Registration Number (TRN / VAT number) printed on the invoice, usually the seller's"),
    ("address", "the primary business address printed on the invoice (usually the seller's)"),
    ("date_of_issue", "the invoice date, exactly as printed"),
    ("invoice_currency", 'the currency code or symbol as printed (e.g. "AED", "USD", "$")'),
    ("amount_excluding_vat", "the subtotal before VAT, exactly as printed - never compute this yourself"),
    ("vat_percent", 'the VAT rate as printed (e.g. "5%")'),
    ("vat_amount", "the VAT amount, exactly as printed - never compute this yourself"),
    ("total_amount", "the total amount due/payable, exactly as printed, including currency symbol"),
    ("beneficiary_name", "the bank account holder name for payment, if printed"),
    ("beneficiary_bank_name", "the beneficiary's bank name, if printed"),
    ("beneficiary_account_number", "the beneficiary's bank account number or IBAN, if printed"),
    ("beneficiary_swift_code", "the beneficiary's SWIFT/BIC code, if printed"),
]
FIELDS = [name for name, _ in FIELD_DEFS]

_schema_lines = []
for _i, (_name, _desc) in enumerate(FIELD_DEFS):
    _schema_lines.append(f'  "{_name}": string or null,  // {_desc}')
    _snippet_comma = "" if _i == len(FIELD_DEFS) - 1 else ","
    _schema_lines.append(f'  "{_name}_snippet": string or null{_snippet_comma}')
_SCHEMA_BLOCK = "\n".join(_schema_lines)

SYSTEM_PROMPT = f"""You are an invoice data extraction engine. You will be shown one image -
one page of a document that may or may not be a tax/commercial invoice. Respond with ONLY
valid JSON (no prose, no markdown fences) matching this schema:

{{
  "document_type": string or null,   // a short label for what this page actually is, using
                                      // its own printed title if it has one (e.g. "Tax Invoice",
                                      // "Delivery Note", "Purchase Order", "Packing List",
                                      // "Credit Note", "Goods Received Note")
  "is_invoice": boolean,             // true ONLY if this page is itself a tax/commercial
                                      // invoice (a bill requesting payment). False for a
                                      // delivery note, purchase order, packing list, goods-
                                      // received note, or any other supporting document -
                                      // even if it prints an invoice number or date for
                                      // cross-reference. When false, output null for every
                                      // field below and its snippet - do not extract values
                                      // from a non-invoice page just because they appear on it.
{_SCHEMA_BLOCK}
}}

Every "*_snippet" field must be the exact, literal, verbatim text copied from the invoice
image that you read the corresponding value from (e.g. the printed line "Total Amount Due:
$90.00" for total_amount, not a paraphrase or the cleaned-up value). This snippet is used to
verify you did not hallucinate the value, so it must be copied character-for-character from
what is visibly printed on the page.

The beneficiary_* fields are the bank/payment details for the party being paid (the invoice
issuer/seller) - extract them only if a payment/bank details block is printed on the invoice.

Rules:
- Never guess or infer a value that is not visibly printed on the invoice.
- If a field is missing, unclear, or not present, output null for BOTH the field and its snippet.
- Do not perform currency conversion or arithmetic - copy amounts, rates, and totals exactly as
  printed, even if you could calculate one from the others.
"""

_model_fields = {"document_type": (Optional[str], None), "is_invoice": (Optional[bool], None)}
for _name in FIELDS:
    _model_fields[_name] = (Optional[str], None)
    _model_fields[f"{_name}_snippet"] = (Optional[str], None)
InvoiceRecord = create_model("InvoiceRecord", **_model_fields)


OUTPUT_COLUMNS = ["source_file", "page", "document_type"] + FIELDS + ["unverified_fields", "format_warnings", "error"]


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_pdf_pages(path: Path) -> list[Image.Image]:
    doc = pdfium.PdfDocument(str(path))
    return [page.render(scale=RENDER_SCALE).to_pil() for page in doc]


def load_pages(path: Path) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        return render_pdf_pages(path)
    return [Image.open(path).convert("RGB")]


def ocr_page_text(image: Image.Image) -> str:
    """Independent, deterministic reading of the page - used only to check the LLM's
    claimed snippets against what is actually printed, never to extract values itself."""
    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:  # noqa: BLE001 - OCR unavailable shouldn't crash extraction
        print(f"[invoice_extract] OCR failed, skipping verification: {exc}", file=sys.stderr)
        return ""


def _normalize(text: str) -> str:
    # Strip whitespace entirely rather than collapsing it: OCR frequently drops or
    # merges spaces around punctuation (e.g. "VAT Rate: 5%" -> "VATRate:5%"), and for
    # short snippets a couple of missing spaces is enough to sink the fuzzy-match ratio.
    return re.sub(r"\s+", "", text.strip().lower())


def _fuzzy_contains(snippet: str, haystack: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """True if `snippet` appears in `haystack` exactly or with high similarity, tolerating
    the OCR noise (0/O, 1/l, spacing) that would otherwise cause false UNVERIFIED flags."""
    snippet_norm = _normalize(snippet)
    haystack_norm = _normalize(haystack)
    if not snippet_norm:
        return False
    if snippet_norm in haystack_norm:
        return True
    window = len(snippet_norm)
    step = max(1, window // 2)
    for start in range(0, max(1, len(haystack_norm) - window + 1), step):
        chunk = haystack_norm[start : start + window]
        if SequenceMatcher(None, snippet_norm, chunk).ratio() >= threshold:
            return True
    return False


def verify_snippets(record: InvoiceRecord, ocr_text: str) -> list[str]:
    """Return field names whose value's snippet could not be found in the page's OCR
    text - i.e. the LLM either hallucinated the value or misreported its own evidence."""
    unverified = []
    for field in FIELDS:
        value = getattr(record, field)
        snippet = getattr(record, f"{field}_snippet")
        if value is None:
            continue
        if not snippet or not _fuzzy_contains(snippet, ocr_text):
            unverified.append(field)
    return unverified


def _parse_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        return None


# Identifier-style fields that should always be plain ASCII per invoicing/banking
# convention (SWIFT codes, TRNs, invoice numbers, account numbers). A non-ASCII
# character here is essentially always wrong - most commonly a vision model reading a
# Latin letter correctly but encoding a look-alike character from another script (e.g.
# Greek "Τ" U+03A4 instead of Latin "T"), which is invisible to the eye but breaks any
# exact-match lookup downstream and is easily missed by fuzzy OCR verification since
# the edit distance is tiny. Not applied to name/address fields, which can legitimately
# contain non-Latin scripts.
ASCII_ONLY_FIELDS = ["tax_invoice_no", "trn", "beneficiary_account_number", "beneficiary_swift_code"]


def format_warnings(record: InvoiceRecord) -> list[str]:
    """Cheap, deterministic sanity checks: obviously-wrong field shapes (e.g. a
    description returned instead of a number), non-ASCII characters in fields that
    should always be plain ASCII, and a VAT arithmetic identity check
    (amount_excluding_vat + vat_amount ~= total_amount) - independent of the OCR
    cross-check, which can't catch a value that's real but simply the wrong one."""
    warnings = []
    for field in ("total_amount", "amount_excluding_vat", "vat_amount", "vat_percent"):
        value = getattr(record, field)
        if value and not re.search(r"\d", value):
            warnings.append(field)
    if record.date_of_issue and not re.search(r"\d", record.date_of_issue):
        warnings.append("date_of_issue")
    if record.tax_invoice_no and not re.search(r"[A-Za-z0-9]", record.tax_invoice_no):
        warnings.append("tax_invoice_no")
    for field in ASCII_ONLY_FIELDS:
        value = getattr(record, field)
        if value and re.search(r"[^\x00-\x7F]", value):
            warnings.append(f"{field}_non_ascii")

    excl_vat = _parse_amount(record.amount_excluding_vat)
    vat = _parse_amount(record.vat_amount)
    total = _parse_amount(record.total_amount)
    if excl_vat is not None and vat is not None and total is not None:
        if abs((excl_vat + vat) - total) > max(0.01, total * 0.01):
            warnings.append("vat_arithmetic_mismatch")
    elif excl_vat is not None and record.vat_percent is None and record.vat_amount is None:
        # No VAT rate or VAT amount was found anywhere on the invoice, so there's no
        # genuine "excluding VAT" figure to report either - this combination usually
        # means the model duplicated total_amount into this field rather than reading
        # an actual VAT breakdown, which the OCR-snippet check can't catch (the
        # duplicated number really is printed on the page - just not as this field).
        warnings.append("amount_excluding_vat_without_vat_breakdown")

    return warnings


def extract_invoice(image: Image.Image, model: str) -> Optional[InvoiceRecord]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the invoice fields from this image."},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
            ],
        },
    ]
    last_error: Optional[Exception] = None
    for _attempt in range(2):
        try:
            response = litellm.completion(
                model=model, messages=messages, temperature=0, timeout=60, max_tokens=1024
            )
            raw = response["choices"][0]["message"]["content"]
            data = json.loads(_strip_fences(raw))
            return InvoiceRecord.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Your previous response failed validation: {exc}\n"
                    "Respond again with ONLY valid JSON matching the schema.",
                }
            )
        except Exception as exc:  # noqa: BLE001 - network/provider errors, don't retry
            last_error = exc
            break
    print(f"[invoice_extract] extraction failed after retries: {last_error}", file=sys.stderr)
    return None


def process_files(paths: list[Path], model: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            pages = load_pages(path)
        except Exception as exc:
            print(f"[invoice_extract] could not open {path}: {exc}", file=sys.stderr)
            rows.append({"source_file": path.name, "page": None, "error": str(exc)})
            continue
        for i, image in enumerate(pages, start=1):
            record = extract_invoice(image, model)
            row = {"source_file": path.name, "page": i}
            if record is None:
                row["error"] = "extraction_failed"
            elif not record.is_invoice:
                # Deterministic override: even if the model filled in fields anyway
                # (e.g. a delivery note repeating the invoice number for cross-reference),
                # never emit them - that would create a phantom duplicate invoice row.
                row["document_type"] = record.document_type or "not an invoice"
                print(f"[invoice_extract] {path.name} page {i}: {row['document_type']} - skipped")
                rows.append(row)
                continue
            else:
                row.update({k: v for k, v in record.model_dump().items() if k in FIELDS + ["document_type"]})
                ocr_text = ocr_page_text(image)
                row["unverified_fields"] = ", ".join(verify_snippets(record, ocr_text))
                row["format_warnings"] = ", ".join(format_warnings(record))
            rows.append(row)
            print(f"[invoice_extract] {path.name} page {i}: {'ok' if record else 'FAILED'}")
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="PDF or image files to process")
    parser.add_argument("--out", default="invoices.csv", help="Output path (.csv or .xlsx)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"litellm model string (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        parser.error(f"file(s) not found: {', '.join(str(p) for p in missing)}")

    df = process_files(paths, args.model)

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".xlsx":
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
