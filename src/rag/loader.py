"""Extract text from PDF documents with pypdf."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def load_pdf(pdf_path: str | Path) -> list[dict]:
    """Return one entry per page, including pages without extractable text.

    Each entry has the shape ``{"page": int, "text": str}``.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path.name}")

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ValueError(
            f"Could not read '{path.name}': the file is not a valid PDF"
        ) from exc

    if reader.is_encrypted:
        raise ValueError(
            f"Could not read '{path.name}': the PDF is encrypted"
        )

    pages = []

    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })
    except PdfReadError as exc:
        raise ValueError(
            f"Could not read '{path.name}': the PDF is malformed"
        ) from exc

    return pages
