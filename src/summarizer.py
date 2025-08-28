"""
Document summarization module using T5 transformer model
"""
from typing import Optional
import re

# Try to import AI libraries, use fallback if not available
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
    torch = None


class DocumentSummarizer:
    """Class to handle document summarization using T5 model"""
    
    def __init__(self):
        self.model_name = "t5-small"  # Using smaller model for faster inference
        self.tokenizer = None
        self.model = None
        self.has_ai_libs = HAS_AI_LIBS
        if HAS_AI_LIBS:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.max_input_length = 512
        self.max_summary_length = 150
    
    def load_model(self):
        """Load T5 model and tokenizer"""
        if not self.has_ai_libs:
            return False
            
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading T5 model: {str(e)}")
            return False
    
    def summarize_text(self, text: str, max_length: int = 150) -> Optional[str]:
        """
        Generate summary of the input text using T5 model or fallback method
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            str: Generated summary or None if summarization fails
        """
        # Use fallback method if AI libraries aren't available
        if not self.has_ai_libs:
            return self._fallback_summarize(text, max_length)
            
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return self._fallback_summarize(text, max_length)
        
        try:
            # Preprocess text for T5 (add task prefix)
            input_text = f"summarize: {text}"
            
            # Tokenize input text
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode the summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return self._fallback_summarize(text, max_length)
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """
        Split long text into smaller chunks for processing
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _fallback_summarize(self, text: str, max_length: int = 150) -> Optional[str]:
        """
        Simple fallback summarization using extractive approach
        
        Args:
            text: Input text to summarize
            max_length: Target number of words for summary
            
        Returns:
            str: Extracted summary
        """
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return "Unable to generate summary."
            
            # Simple scoring: prefer longer sentences and those with common words
            scored_sentences = []
            for sentence in sentences:
                words = sentence.split()
                # Simple scoring based on sentence length and position
                score = len(words) * 0.5
                if sentences.index(sentence) < 3:  # Early sentences often important
                    score += 10
                scored_sentences.append((score, sentence))
            
            # Sort by score and take top sentences
            scored_sentences.sort(reverse=True)
            
            # Build summary up to word limit
            summary_sentences = []
            word_count = 0
            target_words = min(max_length, 200)  # Cap at reasonable length
            
            for score, sentence in scored_sentences:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= target_words:
                    summary_sentences.append(sentence)
                    word_count += sentence_words
                if word_count >= target_words * 0.8:  # 80% of target is enough
                    break
            
            if not summary_sentences:
                # Fallback: take first few sentences
                summary_sentences = sentences[:min(3, len(sentences))]
            
            summary = '. '.join(summary_sentences)
            if not summary.endswith('.'):
                summary += '.'
                
            return summary
            
        except Exception as e:
            print(f"Error in fallback summarization: {str(e)}")
            return "Unable to generate summary due to an error."
    
    def summarize_long_text(self, text: str, max_length: int = 150) -> Optional[str]:
        """
        Summarize long text by chunking and then summarizing the summaries
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of final summary
            
        Returns:
            str: Generated summary or None if summarization fails
        """
        try:
            # If text is short enough, summarize directly
            if len(text) <= 3000:
                return self.summarize_text(text, max_length)
            
            # Chunk the text
            chunks = self.chunk_text(text, chunk_size=2000)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                summary = self.summarize_text(chunk, max_length=100)
                if summary:
                    chunk_summaries.append(summary)
            
            if not chunk_summaries:
                return self._fallback_summarize(text, max_length)
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # If combined summary is still long, summarize it again
            if len(combined_summary) > 2000:
                final_summary = self.summarize_text(combined_summary, max_length)
            else:
                final_summary = combined_summary
            
            return final_summary
            
        except Exception as e:
            print(f"Error summarizing long text: {str(e)}")
            return self._fallback_summarize(text, max_length)
