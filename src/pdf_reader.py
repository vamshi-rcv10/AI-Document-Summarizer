"""
PDF text extraction module using PyPDF2
"""
import PyPDF2
from io import BytesIO
from typing import Optional


class PDFReader:
    """Class to handle PDF text extraction"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text from PDF or None if extraction fails
        """
        try:
            # Read the PDF file
            pdf_bytes = BytesIO(pdf_file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            # Clean up the text
            text = self._clean_text(text)
            
            if not text.strip():
                return None
                
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize line breaks
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()
    
    def validate_pdf(self, pdf_file) -> bool:
        """
        Validate if the uploaded file is a valid PDF
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            bool: True if valid PDF, False otherwise
        """
        try:
            if pdf_file.type != "application/pdf":
                return False
                
            # Try to read the PDF structure
            pdf_bytes = BytesIO(pdf_file.read())
            PyPDF2.PdfReader(pdf_bytes)
            
            # Reset file pointer for later use
            pdf_file.seek(0)
            
            return True
            
        except Exception:
            return False
