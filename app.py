"""
AI-Powered Document Analysis Tool
Main Streamlit application for document summarization and Q&A
"""
import streamlit as st
import os
from src.pdf_reader import PDFReader
from src.summarizer import DocumentSummarizer
from src.qna_bot import QnABot


def initialize_session_state():
    """Initialize session state variables"""
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []


def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="AI Document Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🤖 AI-Powered Document Analysis Tool")
    st.markdown("Upload a PDF document to generate summaries and ask questions using advanced AI models.")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document for analysis"
        )
        
        if uploaded_file is not None:
            # Initialize PDF reader
            pdf_reader = PDFReader()
            
            # Validate PDF
            if not pdf_reader.validate_pdf(uploaded_file):
                st.error("❌ Invalid PDF file. Please upload a valid PDF document.")
                return
            
            # Check if this is a new document
            if (st.session_state.document_name != uploaded_file.name or 
                st.session_state.document_text is None):
                
                with st.spinner("📖 Extracting text from PDF..."):
                    # Extract text from PDF
                    document_text = pdf_reader.extract_text_from_pdf(uploaded_file)
                    
                    if document_text:
                        st.session_state.document_text = document_text
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.summary = None  # Reset summary for new document
                        st.session_state.qa_history = []  # Reset Q&A history
                        st.success(f"✅ Successfully extracted text from {uploaded_file.name}")
                        st.info(f"📊 Document length: {len(document_text)} characters")
                    else:
                        st.error("❌ Failed to extract text from PDF. The document might be scanned or corrupted.")
                        return
        
        # Document info
        if st.session_state.document_text:
            st.markdown("---")
            st.markdown("**📋 Current Document:**")
            st.write(f"📄 {st.session_state.document_name}")
            st.write(f"📝 {len(st.session_state.document_text)} characters")
            
            # Clear document button
            if st.button("🗑️ Clear Document"):
                st.session_state.document_text = None
                st.session_state.document_name = None
                st.session_state.summary = None
                st.session_state.qa_history = []
                st.rerun()
    
    # Main content area
    if st.session_state.document_text:
        # Create tabs for different features
        tab1, tab2, tab3 = st.tabs(["📄 Document Preview", "📝 Summary", "❓ Q&A"])
        
        with tab1:
            st.header("📄 Document Preview")
            
            # Show document preview
            preview_length = min(1000, len(st.session_state.document_text))
            st.text_area(
                "Document Text (First 1000 characters)",
                value=st.session_state.document_text[:preview_length] + 
                      ("..." if len(st.session_state.document_text) > preview_length else ""),
                height=300,
                disabled=True
            )
            
            # Document statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Characters", len(st.session_state.document_text))
            with col2:
                word_count = len(st.session_state.document_text.split())
                st.metric("Word Count", word_count)
            with col3:
                paragraph_count = len([p for p in st.session_state.document_text.split('\n') if p.strip()])
                st.metric("Paragraphs", paragraph_count)
            with col4:
                estimated_reading_time = max(1, word_count // 200)  # Average reading speed
                st.metric("Est. Reading Time", f"{estimated_reading_time} min")
        
        with tab2:
            st.header("📝 Document Summary")
            
            # Summary controls
            col1, col2 = st.columns([3, 1])
            with col1:
                summary_length = st.slider(
                    "Summary Length",
                    min_value=50,
                    max_value=300,
                    value=150,
                    step=25,
                    help="Adjust the maximum length of the generated summary"
                )
            with col2:
                generate_summary = st.button("🔄 Generate Summary", type="primary")
            
            # Generate or display summary
            if generate_summary or (st.session_state.summary is None):
                with st.spinner("🤖 Generating summary using T5 model..."):
                    summarizer = DocumentSummarizer()
                    summary = summarizer.summarize_long_text(
                        st.session_state.document_text,
                        max_length=summary_length
                    )
                    
                    if summary:
                        st.session_state.summary = summary
                    else:
                        st.error("❌ Failed to generate summary. Please try again.")
            
            # Display summary
            if st.session_state.summary:
                st.subheader("📋 Generated Summary")
                st.write(st.session_state.summary)
                
                # Summary statistics
                summary_word_count = len(st.session_state.summary.split())
                original_word_count = len(st.session_state.document_text.split())
                compression_ratio = (1 - summary_word_count / original_word_count) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Summary Words", summary_word_count)
                with col2:
                    st.metric("Original Words", original_word_count)
                with col3:
                    st.metric("Compression", f"{compression_ratio:.1f}%")
        
        with tab3:
            st.header("❓ Question & Answer")
            
            # Q&A interface
            question = st.text_input(
                "Ask a question about the document:",
                placeholder="e.g., What is the main topic of this document?",
                help="Enter your question and the AI will try to find the answer in the document"
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                ask_question = st.button("🔍 Ask Question", type="primary", disabled=not question.strip())
            
            # Process question
            if ask_question and question.strip():
                with st.spinner("🤖 Searching for answer using BERT model..."):
                    qna_bot = QnABot()
                    result = qna_bot.answer_question_with_context_search(
                        question.strip(),
                        st.session_state.document_text
                    )
                    
                    if result:
                        # Add to history
                        st.session_state.qa_history.append({
                            "question": question.strip(),
                            "answer": result["answer"],
                            "confidence": result["confidence"]
                        })
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.write(result["answer"])
                        
                        # Confidence indicator
                        confidence_percentage = result["confidence"] * 100
                        if confidence_percentage >= 50:
                            confidence_color = "green"
                            confidence_emoji = "🟢"
                        elif confidence_percentage >= 25:
                            confidence_color = "orange"
                            confidence_emoji = "🟡"
                        else:
                            confidence_color = "red"
                            confidence_emoji = "🔴"
                        
                        st.markdown(
                            f"{confidence_emoji} **Confidence:** "
                            f"<span style='color: {confidence_color}'>{confidence_percentage:.1f}%</span>",
                            unsafe_allow_html=True
                        )
                        
                        # Show context if available
                        if "context_used" in result:
                            with st.expander("📄 Context Used"):
                                st.write(result["context_used"])
                    else:
                        st.error("❌ Failed to process question. Please try again.")
            
            # Q&A History
            if st.session_state.qa_history:
                st.markdown("---")
                st.subheader("📚 Question History")
                
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(st.session_state.qa_history) - i}: {qa['question'][:50]}..."):
                        st.write(f"**Question:** {qa['question']}")
                        st.write(f"**Answer:** {qa['answer']}")
                        st.write(f"**Confidence:** {qa['confidence']*100:.1f}%")
                
                # Clear history button
                if st.button("🗑️ Clear Q&A History"):
                    st.session_state.qa_history = []
                    st.rerun()
    
    else:
        # Welcome screen when no document is uploaded
        st.markdown("""
        ## 👋 Welcome to the AI Document Analyzer!
        
        This tool uses state-of-the-art AI models to help you analyze your documents:
        
        ### 🚀 Features:
        - **📄 PDF Text Extraction**: Extract text from PDF documents automatically
        - **📝 AI Summarization**: Generate concise summaries using T5 transformer model
        - **❓ Question Answering**: Ask questions and get answers using BERT model
        - **📊 Document Statistics**: Get insights about your document's content
        
        ### 🛠️ How to Use:
        1. **Upload a PDF** document using the sidebar
        2. **Preview** the extracted text in the Document Preview tab
        3. **Generate a summary** in the Summary tab
        4. **Ask questions** about the document in the Q&A tab
        
        ### 🤖 AI Models Used:
        - **T5** (Text-To-Text Transfer Transformer) for summarization
        - **BERT** (Bidirectional Encoder Representations from Transformers) for Q&A
        
        ---
        
        📁 **To get started, upload a PDF document using the sidebar!**
        """)
        
        # Additional tips
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - **Document Quality**: Clear, well-formatted PDFs work best
            - **Question Style**: Ask specific questions about the content
            - **Summary Length**: Adjust summary length based on document size
            - **Processing Time**: Larger documents may take longer to process
            """)


if __name__ == "__main__":
    main()
