import pytest
import pandas as pd
from src.preprocess import clean_text  # Assume you put your Task 1 logic here
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_clean_text():
    sample = "I am writing to file a complaint... [XX/XX/XXXX] regarding my Credit Card."
    cleaned = clean_text(sample)
    assert "writing to file a complaint" not in cleaned
    assert "[XX/XX/XXXX]" not in cleaned
    assert cleaned == cleaned.lower()

def test_chunking_logic():
    text = "This is a very long sentence used to test if the chunker correctly splits the narrative into smaller parts for the vector store."
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks) # Allow some margin for overlap