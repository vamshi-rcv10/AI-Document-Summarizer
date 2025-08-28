# AI-Powered Document Analysis Tool

## Overview

This is a Streamlit-based web application that provides AI-powered document analysis capabilities. The tool allows users to upload PDF documents and perform two main operations: automatic text summarization and interactive question-answering. The application leverages pre-trained transformer models (T5 for summarization and DistilBERT for Q&A) to provide intelligent document analysis without requiring external API calls or cloud services.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Session Management**: Streamlit's built-in session state for maintaining user data across interactions
- **UI Components**: Sidebar for file upload, main area for displaying results and interactions
- **Layout**: Wide layout with expandable sidebar for better user experience

### Backend Architecture
- **Modular Design**: Separation of concerns with dedicated modules for different functionalities:
  - `PDFReader`: Handles PDF text extraction using PyPDF2
  - `DocumentSummarizer`: Manages text summarization using T5 transformer model
  - `QnABot`: Provides question-answering capabilities using DistilBERT model
- **Model Management**: Each AI component handles its own model loading and inference
- **Error Handling**: Graceful degradation with try-catch blocks for model operations
- **Memory Optimization**: Uses smaller pre-trained models (t5-small, distilbert-base-cased-distilled-squad) for faster inference

### AI/ML Architecture
- **Summarization Strategy**: T5 (Text-to-Text Transfer Transformer) model for abstractive summarization
- **Q&A Strategy**: DistilBERT model fine-tuned on SQuAD dataset for extractive question answering
- **Device Detection**: Automatic GPU/CPU detection with fallback to CPU processing
- **Text Processing**: Token length limits and text cleaning for optimal model performance

### Data Flow
- **Document Processing Pipeline**: PDF upload → Text extraction → Model inference → Result display
- **Session Persistence**: Document text, summaries, and Q&A history maintained across user interactions
- **Streaming Interface**: Real-time feedback and progressive loading for better user experience

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **PyPDF2**: PDF text extraction and processing
- **Transformers (Hugging Face)**: Pre-trained transformer models for NLP tasks
- **PyTorch**: Deep learning framework for model inference

### Pre-trained Models
- **T5-Small**: Google's Text-to-Text Transfer Transformer for summarization tasks
- **DistilBERT-base-cased-distilled-squad**: Lightweight BERT model for question answering
- **Hugging Face Model Hub**: Source for downloading and loading pre-trained models

### System Dependencies
- **CUDA (Optional)**: GPU acceleration for faster model inference when available
- **Python Standard Library**: BytesIO for file handling, typing for type hints

Note: This application is designed to run entirely offline once models are downloaded, with no external API dependencies or cloud service requirements.