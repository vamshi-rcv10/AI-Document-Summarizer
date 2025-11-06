"""
Improved Document summarizer using T5 transformer model
- Better input formatting
- Structured summaries
- Cleaner fallback summarizer
"""

from typing import Optional
import re

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
    torch = None


class DocumentSummarizer:
    """Class to summarize documents using T5"""
    
    def __init__(self):
        self.model_name = "t5-base"  # <- BETTER MODEL THAN t5-small
        self.tokenizer = None
        self.model = None
        self.has_ai_libs = HAS_AI_LIBS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_AI_LIBS else None
        self.max_input_length = 1024
        self.max_summary_length = 250
    
    def load_model(self):
        if not self.has_ai_libs:
            return False
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            return True

        except Exception as e:
            print(f"Error loading T5 model: {e}")
            return False
    
    def summarize_text(self, text: str, max_length: int = 200) -> Optional[str]:
        if not self.has_ai_libs:
            return self._fallback_summarize(text, max_length)

        if not self.model or not self.tokenizer:
            if not self.load_model():
                return self._fallback_summarize(text, max_length)

        try:
            prompt = (
                "summarize this document into structured key points. "
                "Focus on important information, remove filler text:\n\n"
                f"{text}"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=80,
                    num_beams=6,              # <- MORE BEAMS = BETTER QUALITY
                    no_repeat_ngram_size=3,   # <- Prevent repetition
                    early_stopping=True
                )

            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            print(f"Error summarizing text: {e}")
            return self._fallback_summarize(text, max_length)

    def _fallback_summarize(self, text: str, max_length: int = 200) -> Optional[str]:
        """Simple extractive summarization fallback"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        if not sentences:
            return "Unable to generate summary."

        top_sentences = sentences[:4]  # First 4 sentences usually contain core context
        return "\n- " + "\n- ".join(top_sentences)

    def summarize_long_text(self, text: str, max_length: int = 200) -> Optional[str]:
        """Summarize large documents in chunks"""
        if len(text) < 3000:
            return self.summarize_text(text, max_length)

        chunks = [text[i:i + 3000] for i in range(0, len(text), 3000)]

        partial_summaries = []
        for chunk in chunks:
            summary = self.summarize_text(chunk, max_length=120)
            partial_summaries.append(summary)

        combined = "\n".join(partial_summaries)
        return self.summarize_text(combined, max_length)
