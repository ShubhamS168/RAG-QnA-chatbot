import re
from io import BytesIO
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf(file: BytesIO) -> str:
    """
    Reads text from a PDF file.

    Args:
        file (BytesIO): The PDF file object.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text

def read_txt(file: BytesIO) -> str:
    """
    Reads text from a TXT file.

    Args:
        file (BytesIO): The TXT file object.

    Returns:
        str: Extracted text from the TXT.
    """
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def read_docx(file: BytesIO) -> str:
    """
    Reads text from a DOCX file.

    Args:
        file (BytesIO): The DOCX file object.

    Returns:
        str: Extracted text from the DOCX.
    """
    text = ""
    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""
    return text

def read_csv(file: BytesIO) -> str:
    """
    Reads text from a CSV file, converting it into a string representation.

    Args:
        file (BytesIO): The CSV file object.

    Returns:
        str: String representation of the CSV content.
    """
    try:
        df = pd.read_csv(file)
        # Convert DataFrame to a string, useful for RAG
        return df.to_string(index=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return ""

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning.

    Args:
        text (str): The input text.

    Returns:
        str: Cleaned text.
    """
    # Remove multiple newlines and replace with single newline
    text = re.sub(r'\n+', '\n', text)
    # Remove multiple spaces and replace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits a large text into smaller, overlapping chunks.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []
