import re
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30):
        """
        Handles document chunking with a sliding window approach.
        
        :param chunk_size: Approximate word target per text block.
        :param chunk_overlap: Number of words to carry over to the next chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Splits raw text into overlapping paragraphs based on word boundaries."""
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split(' ')
        
        if len(words) <= self.chunk_size:
            return [text] if text else []
            
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = start_idx + self.chunk_size
            chunk_words = words[start_idx:end_idx]
            chunks.append(" ".join(chunk_words))
            
            start_idx += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def process_pdf(self, pdf_file) -> List[Dict[str, any]]:
        """Parses an uploaded PDF file and builds metadata-mapped chunk payloads."""
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_file)
        processed_payloads = []
        
        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            page_chunks = self.split_text_into_chunks(raw_text)
            
            for chunk in page_chunks:
                processed_payloads.append({
                    "text": chunk,
                    "metadata": {
                        "page_number": page_idx + 1
                    }
                })
                
        return processed_payloads