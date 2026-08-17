"""Streamlit UI for invoice_extract.py.

Upload invoice PDFs/images, watch per-page extraction progress, view results
as a table, and download as CSV or Excel. Standalone from the financial
statement pipeline's app.py.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from invoice_extract import (
    DEFAULT_MODEL,
    FIELDS,
    OUTPUT_COLUMNS,
    extract_invoice,
    format_warnings,
    load_pages,
    ocr_page_text,
    verify_snippets,
)

st.set_page_config(page_title="Invoice Extraction", layout="wide")
st.title("Invoice Data Extraction")
st.caption(
    "Upload one or more invoice PDFs or images (scanned/photographed is fine). "
    "Each PDF page is treated as one invoice."
)

uploaded_files = st.file_uploader(
    "Upload invoice PDF(s) or image(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

model = st.text_input("Model (litellm string)", value=DEFAULT_MODEL)

if uploaded_files and st.button("Extract", type="primary"):
    tmp_dir = Path(tempfile.mkdtemp())
    rows = []
    with st.status("Extracting invoices...", expanded=True) as status:
        for uploaded in uploaded_files:
            dest = tmp_dir / uploaded.name
            dest.write_bytes(uploaded.getbuffer())
            try:
                pages = load_pages(dest)
            except Exception as exc:  # noqa: BLE001 - surface any file-open failure and continue
                st.write(f":x: {uploaded.name}: could not open ({exc})")
                rows.append({"source_file": uploaded.name, "page": None, "error": str(exc)})
                continue
            for i, image in enumerate(pages, start=1):
                st.write(f"Processing {uploaded.name} page {i}/{len(pages)}...")
                record = extract_invoice(image, model)
                row = {"source_file": uploaded.name, "page": i}
                if record is None:
                    row["error"] = "extraction_failed"
                    st.write(f":warning: {uploaded.name} page {i}: extraction failed")
                elif not record.is_invoice:
                    # Deterministic override: never emit invoice fields for a page the
                    # model itself says isn't an invoice, even if it filled some in anyway
                    # (e.g. a delivery note repeating the invoice number for cross-reference).
                    row["document_type"] = record.document_type or "not an invoice"
                    st.write(f":information_source: {uploaded.name} page {i}: {row['document_type']} - skipped")
                else:
                    row.update({k: v for k, v in record.model_dump().items() if k in FIELDS + ["document_type"]})
                    ocr_text = ocr_page_text(image)
                    unverified = verify_snippets(record, ocr_text)
                    row["unverified_fields"] = ", ".join(unverified)
                    row["format_warnings"] = ", ".join(format_warnings(record))
                    flag = f" :warning: unverified: {row['unverified_fields']}" if unverified else ""
                    st.write(f":white_check_mark: {uploaded.name} page {i}: done{flag}")
                rows.append(row)
        status.update(label="Done", state="complete")

    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    st.session_state["invoice_df"] = df[OUTPUT_COLUMNS]

if "invoice_df" in st.session_state:
    df = st.session_state["invoice_df"]
    flagged = df["unverified_fields"].fillna("").apply(bool) if "unverified_fields" in df else pd.Series([False] * len(df))
    if flagged.any():
        st.warning(
            f":warning: {flagged.sum()} of {len(df)} row(s) have a value whose snippet "
            "couldn't be matched against the page's OCR text - review these before trusting them."
        )

    st.subheader("Extracted invoices")

    def _highlight_unverified(row: pd.Series) -> list[str]:
        if row.get("unverified_fields"):
            return ["background-color: #4a2020" for _ in row]
        is_skipped = pd.isna(row.get("tax_invoice_no")) and pd.notna(row.get("document_type")) and pd.isna(row.get("error"))
        if is_skipped:
            return ["background-color: #2a2a2a; color: #888888" for _ in row]
        return ["" for _ in row]

    st.dataframe(df.style.apply(_highlight_unverified, axis=1), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="invoices.csv", mime="text/csv")

    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False)
    st.download_button(
        "Download Excel",
        excel_buf.getvalue(),
        file_name="invoices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
