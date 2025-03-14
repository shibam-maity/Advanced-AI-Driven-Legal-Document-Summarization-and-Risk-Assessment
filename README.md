# 📄 AI-Driven Legal Document Analysis System



A powerful, AI-driven tool designed to analyze, summarize, and assess risks in legal documents. Leveraging state-of-the-art language models and specialized legal analysis techniques to help legal professionals save time and gain deeper insights.



## ✨ Key Features

- 📃 **Document Summarization**: Generate concise, accurate summaries of complex legal documents
- 🔍 **Risk Assessment**: Identify potential legal risks with severity ratings and visualizations
- 💬 **Interactive Q&A**: Ask questions about the document and receive contextual answers
- 🔀 **Document Comparison**: Compare two legal documents with detailed difference analysis and tabular views
- 📋 **Compliance Analysis**: Identify relevant regulatory requirements for specific document types
- 📊 **Visual Reports**: Generate comprehensive PDF reports with visualizations
- 📧 **Email Integration**: Send analysis reports directly via email
- 📜 **Legal Updates**: Stay informed about relevant legal changes related to your documents

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **Language Models**: LangChain + Groq (Llama 3)
- **Document Processing**: PyMuPDF, NLTK, Regex
- **Vector Search**: FAISS, Sentence Transformers
- **Data Visualization**: Plotly, Pandas
- **PDF Generation**: FPDF
- **Email Service**: SendGrid
- **Web Scraping**: BeautifulSoup, Requests

## 🏗️ Architecture

The system follows a modular architecture with specialized components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Document       │────▶│  Analysis       │────▶│  Visualization  │
│  Processing     │     │  Engine         │     │  & Reporting    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Vector         │     │  Legal          │     │  Email          │
│  Database       │     │  Knowledge Base │     │  Service        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🔧 Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/legal-document-analysis.git
   cd legal-document-analysis
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys
   ```
   GROQ_API_KEY=your_groq_api_key
   SENDGRID_API_KEY=your_sendgrid_api_key
   SENDER_EMAIL=your_sender_email
   ```

5. Run the application
   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```

## 📋 Usage Guide

### Document Analysis
1. Upload your legal document (PDF format)
2. Click "Analyze Document" to process
3. View the generated summary and navigate to other tabs for detailed analysis

### Risk Assessment
- The Risk Dashboard provides visual representations of identified risks
- Risks are categorized by severity (Critical, High, Medium, Low)
- Interactive charts show risk distribution by category

### Document Q&A
- Ask specific questions about the document content
- The system uses RAG (Retrieval Augmented Generation) to provide accurate answers
- Previous questions and answers are saved in chat history

### Document Comparison
- Upload a second document to compare with your original document
- View differences highlighted in an interactive display
- Choose between detailed comparison or tabular comparison views
- Generate a comparison report for sharing

### Compliance Analysis
- Automatically identifies relevant compliance requirements based on document type
- Shows key regulations, requirements, and recent updates
- Useful for ensuring documents adhere to relevant legal standards

## 🌟 Implementation Highlights

### Advanced Document Processing
The system uses a combination of PyMuPDF for extraction and NLTK for natural language processing to handle complex legal documents with proper structure recognition.

### Semantic Understanding
Instead of simple keyword matching, the system employs semantic embeddings to understand document meaning, enabling more accurate summarization and comparison.

### Retrieval Augmented Generation (RAG)
The Q&A system implements RAG architecture to retrieve relevant document sections before generating answers, ensuring responses are contextually accurate and grounded in the document content.

### Legal-Specific Analysis
Custom-built analyzers for various legal document types (contracts, GDPR documents, employment agreements, etc.) provide specialized insights for each document category.

## 🔮 Future Improvements

- Multi-document analysis and correlation
- Integration with legal case databases
- Collaborative annotations and team workflows
- Support for additional document formats (DOCX, HTML)
- Custom fine-tuning for specific legal domains
- Mobile application version

## ⚠️ Troubleshooting

### PyTorch and Streamlit Compatibility
If you encounter a "RuntimeError: Tried to instantiate class '__path__._path'" error, you have two options:

1. Add this at the top of app.py:
   ```python
   import os
   os.environ["PYTORCH_JIT"] = "0"  # Disable PyTorch JIT
   ```

2. Or run Streamlit with this flag:
   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```

### NLTK Resources
Make sure NLTK resources are properly downloaded:
```python
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
```

## 📜 License

© 2025 VidzAI - All Rights Reserved. This software is proprietary and confidential.



