"""
Question-answering module using BERT transformer model
"""
from typing import Optional, Dict
import re

# Try to import AI libraries, use fallback if not available
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
    torch = None


class QnABot:
    """Class to handle question-answering using BERT model"""
    
    def __init__(self):
        self.model_name = "distilbert-base-cased-distilled-squad"  # Lighter BERT model
        self.tokenizer = None
        self.model = None
        self.has_ai_libs = HAS_AI_LIBS
        if HAS_AI_LIBS:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.max_length = 512
    
    def load_model(self):
        """Load BERT model and tokenizer for question answering"""
        if not self.has_ai_libs:
            return False
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            return False
    
    def answer_question(self, question: str, context: str) -> Optional[Dict]:
        """
        Answer a question based on the provided context using BERT or fallback
        
        Args:
            question: The question to answer
            context: The document text to search for answers
            
        Returns:
            dict: Dictionary with answer, confidence score, and start/end positions
        """
        # Use fallback method if AI libraries aren't available
        if not self.has_ai_libs:
            return self._fallback_answer(question, context)
            
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return self._fallback_answer(question, context)
        
        try:
            # Tokenize question and context
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_offsets_mapping=True
            )
            
            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
            
            # Find the most likely answer span
            start_scores = torch.softmax(start_logits, dim=1)
            end_scores = torch.softmax(end_logits, dim=1)
            
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Calculate confidence score
            confidence = float(start_scores[0][start_idx] * end_scores[0][end_idx])
            
            # Extract answer text
            if start_idx <= end_idx:
                answer_tokens = input_ids[0][start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            else:
                answer = ""
            
            # Clean up the answer
            answer = answer.strip()
            
            # If confidence is too low or answer is empty, indicate uncertainty
            if confidence < 0.1 or not answer:
                return {
                    "answer": "I couldn't find a confident answer to that question in the document.",
                    "confidence": confidence,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx)
                }
            
            return {
                "answer": answer,
                "confidence": confidence,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx)
            }
            
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question: str, context: str) -> Optional[Dict]:
        """
        Simple fallback question-answering using keyword matching
        
        Args:
            question: The question to answer
            context: The document text to search for answers
            
        Returns:
            dict: Dictionary with answer and confidence info
        """
        try:
            # Simple keyword-based answer extraction
            question_lower = question.lower()
            context_sentences = re.split(r'[.!?]+', context)
            context_sentences = [s.strip() for s in context_sentences if s.strip()]
            
            # Extract key words from question (removing common words)
            stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 
                         'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'the', 'a', 'an'}
            question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]
            
            if not question_words:
                return {
                    "answer": "I need more specific keywords to answer your question.",
                    "confidence": 0.1,
                    "start_idx": 0,
                    "end_idx": 0
                }
            
            # Score sentences based on keyword matches
            best_sentence = ""
            best_score = 0
            
            for sentence in context_sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in question_words if word in sentence_lower)
                
                # Boost score for sentences that contain multiple keywords
                if score > 1:
                    score *= 1.5
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            
            # Generate answer
            if best_score > 0 and best_sentence:
                # Try to extract a more specific answer from the sentence
                answer = self._extract_specific_answer(question, best_sentence)
                confidence = min(0.7, best_score / len(question_words))  # Cap confidence at 0.7
                
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "start_idx": 0,
                    "end_idx": len(answer)
                }
            else:
                return {
                    "answer": "I couldn't find a relevant answer to your question in the document.",
                    "confidence": 0.1,
                    "start_idx": 0,
                    "end_idx": 0
                }
                
        except Exception as e:
            print(f"Error in fallback answering: {str(e)}")
            return {
                "answer": "Unable to answer the question due to an error.",
                "confidence": 0.1,
                "start_idx": 0,
                "end_idx": 0
            }
    
    def _extract_specific_answer(self, question: str, sentence: str) -> str:
        """
        Try to extract a more specific answer from a sentence
        
        Args:
            question: The original question
            sentence: The sentence containing the answer
            
        Returns:
            str: Extracted answer or the full sentence
        """
        question_lower = question.lower()
        
        # Simple patterns for common question types
        if question_lower.startswith('what is') or question_lower.startswith('what are'):
            # Look for definitions or descriptions
            return sentence
        elif question_lower.startswith('when'):
            # Look for dates/times
            import re
            dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b', sentence)
            if dates:
                return f"In {dates[0]}"
            return sentence
        elif question_lower.startswith('where'):
            # Look for locations
            return sentence
        elif question_lower.startswith('who'):
            # Look for names/people
            return sentence
        else:
            # For other questions, return the sentence
            return sentence

    def find_relevant_context(self, question: str, text: str, max_context_length: int = 2000) -> str:
        """
        Find the most relevant context from the document for the question
        
        Args:
            question: The question to find context for
            text: The full document text
            max_context_length: Maximum length of context to return
            
        Returns:
            str: Most relevant context chunk
        """
        try:
            # If text is short enough, return as is
            if len(text) <= max_context_length:
                return text
            
            # Split text into overlapping chunks
            chunk_size = max_context_length - 200  # Leave room for overlap
            overlap = 100
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
                if i + chunk_size >= len(text):
                    break
            
            # Score each chunk based on keyword overlap with question
            question_words = set(question.lower().split())
            best_chunk = chunks[0]
            best_score = 0
            
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                score = len(question_words.intersection(chunk_words))
                
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            return best_chunk
            
        except Exception as e:
            print(f"Error finding relevant context: {str(e)}")
            return text[:max_context_length]
    
    def answer_question_with_context_search(self, question: str, document_text: str) -> Optional[Dict]:
        """
        Answer question by first finding relevant context, then using BERT
        
        Args:
            question: The question to answer
            document_text: The full document text
            
        Returns:
            dict: Answer result with confidence and context information
        """
        try:
            # Find relevant context
            relevant_context = self.find_relevant_context(question, document_text)
            
            # Answer question using the relevant context
            result = self.answer_question(question, relevant_context)
            
            if result:
                result["context_used"] = relevant_context[:200] + "..." if len(relevant_context) > 200 else relevant_context
            
            return result
            
        except Exception as e:
            print(f"Error in context search Q&A: {str(e)}")
            return None
