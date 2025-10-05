from pypdf import PdfReader
from typing import Dict, List, Tuple, Optional
import re

def extract_text_from_pdf(file, extract_metadata: bool = True) -> Tuple[str, Optional[Dict]]:
    """Extract text and metadata from a PDF file.
    
    Args:
        file: The PDF file object
        extract_metadata: Whether to extract metadata
        
    Returns:
        Tuple of (extracted text, metadata dict or None)
    """
    reader = PdfReader(file)
    parts = []
    
    # Extract text from each page
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            # Add page number reference
            parts.append(f"[Page {i+1}]\n{text}")
        except Exception:
            parts.append(f"[Page {i+1}]\nText extraction failed for this page.")
    
    # Extract metadata if requested
    metadata = None
    if extract_metadata and reader.metadata:
        metadata = {
            "title": reader.metadata.title,
            "author": reader.metadata.author,
            "subject": reader.metadata.subject,
            "creator": reader.metadata.creator,
            "producer": reader.metadata.producer,
            "page_count": len(reader.pages)
        }
        # Clean up metadata - remove None values and empty strings
        metadata = {k: v for k, v in metadata.items() if v is not None and str(v).strip() != ""}
    
    return "\n\n".join(parts), metadata

def clean_text(text: str) -> str:
    """Clean extracted PDF text by removing excessive whitespace and fixing common issues."""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Fix hyphenated words at line breaks (common in PDFs)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()
