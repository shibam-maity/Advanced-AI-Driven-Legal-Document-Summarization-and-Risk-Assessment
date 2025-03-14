import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import base64
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import difflib
import requests
from bs4 import BeautifulSoup
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

st.set_page_config(
    page_title="LegalDoc Analyst",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize NLP resources
import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data")) # Or your preferred path

try:
    nltk.download('vader_lexicon', download_dir=os.path.join(os.getcwd(), "nltk_data")) #optional
except LookupError as e:
    st.error(f"Error downloading NLTK resources: {e}.  Please check your internet connection and try again.")
    st.stop()

from nltk.sentiment import SentimentIntensityAnalyzer


from document_processing import extract_text_from_pdf
from document_processing import chunk_text, create_faiss_index
from rag import generate_rag_response
from risk_analysis import advanced_risk_assessment, visualize_risks
from summarization import generate_summary
from comparison import compare_documents, export_comparison_report, compare_documents_tabular
from compliance import fetch_updates_for_document, classify_document_type, fetch_document_compliance
from report_generation import generate_pdf, send_email, create_email_text
from utils import initialize_session_state

# Initialize session state
initialize_session_state()


def main():
    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f5f7fb;}
        .stButton>button {border-radius: 8px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {width: 100%;}
        .stExpander .st-emotion-cache-1hynsf2 {border-radius: 10px;}
        .metric-box {padding: 20px; border-radius: 10px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .risk-critical {color: #dc3545!important;}
        .risk-high {color: #ff6b6b!important;}
        .risk-medium {color: #ffd93d!important;}
        .risk-low {color: #6c757d!important;}
        .update-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .update-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .update-source {
            color: #7f8c8d;
            font-size: 0.8rem;
        }
        .update-snippet {
            color: #34495e;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .category-confidence {
            color: #3498db;
            font-weight: bold;
        }
        .compliance-item {
            padding: 10px;
            border-left: 3px solid #3498db;
            background: #f8f9fa;
            margin-bottom: 8px;
            border-radius: 0 4px 4px 0;
        }
        .regulation-item {
            background: #e8f4f8;
            padding: 6px 10px;
            margin-right: 5px;
            margin-bottom: 5px;
            border-radius: 4px;
            display: inline-block;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)

    # App Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    with col2:
        st.title("Advanced AI-Driven Legal Document Summarization and Risk Assessment")
        st.markdown("""**© 2025 VidzAI - All Rights Reserved
                This software is proprietary and confidential. Any unauthorized use, reproduction, or distribution is strictly prohibited.**""")
           
    # Main Layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Document Analysis",
        "📊 Risk Dashboard",
        "❓ Q&A Chat",
        "🔀 Comparison",
        "📜 Compliance",
        "🔍 Legal Updates",
        "📧 Email Report"
    ])

    # Document Processing Section
    with tab1:
        st.header("Document Processing")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])
            if uploaded_file and not st.session_state.document_processed:
                if st.button("Analyze Document", type="primary"):
                    with st.status("Processing document...", expanded=True) as status:
                        try:
                            st.write("Extracting text...")
                            st.session_state.full_text = extract_text_from_pdf(uploaded_file)

                            st.write("Chunking text...")
                            st.session_state.text_chunks = chunk_text(st.session_state.full_text)

                            st.write("Creating search index...")
                            st.session_state.faiss_index = create_faiss_index(st.session_state.text_chunks)

                            st.write("Generating summary...")
                            st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)

                            st.write("Assessing risks...")
                            st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)
                            
                            st.write("Classifying document and fetching updates...")
                            st.session_state.document_categories = classify_document_type(st.session_state.full_text)
                            st.session_state.legal_updates = fetch_updates_for_document(st.session_state.full_text)
                            
                            st.write("Analyzing compliance requirements...")
                            st.session_state.document_compliance = fetch_document_compliance(st.session_state.full_text)

                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            st.session_state.document_processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.session_state.document_processed = False

        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Document Summary")
                st.write(st.session_state.summaries.get('document', "No summary available"))

                # Document classification
                if st.session_state.document_categories:
                    st.subheader("Document Classification")
                    for category, confidence in st.session_state.document_categories[:3]:
                        st.write(f"- {category}: {confidence:.2f} confidence")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "Download Text Summary",
                        data=st.session_state.summaries.get('document', ""),
                        file_name="document_summary.txt",
                        use_container_width=True
                    )
                with col_d2:
                    if st.button("Generate Full Report PDF", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            pdf_buffer = generate_pdf(
                                st.session_state.summaries['document'],
                                st.session_state.risk_data,
                                st.session_state.legal_updates if hasattr(st.session_state, 'legal_updates') else None,
                                st.session_state.document_compliance if hasattr(st.session_state, 'document_compliance') else None
                            )
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("PDF ready for download!")

    # Risk Dashboard
    if st.session_state.document_processed:
        with tab2:
            st.header("Risk Analysis Dashboard")
            risk_data = st.session_state.risk_data

            # Risk Metrics
            with st.container(border=True):
                cols = st.columns(4)
                metric_style = """
                    <style>
                        .metric-box {
                            padding: 20px;
                            border-radius: 10px;
                            background: white;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            height: 150px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        }
                        .metric-title {
                            font-size: 1.1rem;
                            margin-bottom: 8px;
                            font-weight: 600;
                            color: #666;
                        }
                        .metric-value {
                            font-size: 2.5rem;
                            font-weight: 700;
                            line-height: 1.2;
                            color: #dc3545 !important;
                        }
                        .metric-subtext {
                            font-size: 1rem;
                            color: #666;
                        }
                        .risk-critical { color: #dc3545; }
                        .risk-high { color: #ff6b6b; }
                    </style>
                """
                st.markdown(metric_style, unsafe_allow_html=True)

                # Overall Risk Score
                with cols[0]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-critical">Overall Risk Score</div>
                            <div class="metric-value risk-critical">
                                {risk_data.get("total_score", 0)}
                                <span class="metric-subtext">/100</span>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # Total Risks
                with cols[1]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title">Total Risks</div>
                            <div class="metric-value">
                                {risk_data.get("total_risks", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # Critical Risks
                with cols[2]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-critical">Critical Risks</div>
                            <div class="metric-value risk-critical">
                                {risk_data["severity_counts"].get("Critical", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # High Risks
                with cols[3]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-high">High Risks</div>
                            <div class="metric-value risk-high">
                                {risk_data["severity_counts"].get("High", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

            # Visualizations
            with st.container(border=True):
                fig1, fig2 = visualize_risks(risk_data)
                if fig1 and fig2:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col_v2:
                        st.plotly_chart(fig2, use_container_width=True)

            # Detailed Risk Breakdown
            with st.container(border=True):
                st.subheader("Risk Category Breakdown")
                if risk_data.get('categories'):
                    df = pd.DataFrame.from_dict(risk_data['categories'], orient='index')
                    st.dataframe(
                        df,
                        column_config={
                            "score": st.column_config.ProgressColumn(
                                "Score",
                                help="Risk score (0-40)",
                                format="%f",
                                min_value=0,
                                max_value=40,
                            )
                        },
                        use_container_width=True
                    )

    # Chat Interface (Now as tab3)
    with tab3:
        st.header("Document Q&A")
        if st.session_state.document_processed:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for role, msg in st.session_state.chat_history:
                    with st.chat_message(role):
                        st.write(msg)
            
            # Input for new questions
            st.write("---")
            query = st.chat_input("Ask about the document...")
            if query:
                with st.spinner("Analyzing..."):
                    response = generate_rag_response(query, st.session_state.faiss_index, st.session_state.text_chunks)
                    st.session_state.chat_history.extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()
        else:
            st.info("Please upload and analyze a document first to use the Q&A feature.")
            
            # Sample Q&A to show functionality
            with st.expander("Sample Q&A"):
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <h4 style="margin-top: 0;">Example Questions You Can Ask:</h4>
                    <ul>
                        <li>What are the key obligations in this contract?</li>
                        <li>Explain the termination clause in simple terms.</li>
                        <li>What are the payment terms?</li>
                        <li>Are there any concerning liability clauses?</li>
                        <li>Summarize the confidentiality requirements.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Document Comparison (Now as tab4)
    with tab4:
        st.header("Document Comparison")
        if st.session_state.document_processed:
            with st.container(border=True):
                compare_file = st.file_uploader("Upload Comparison Document", type=["pdf"])
                if compare_file:
                    try:
                        # Add document names for better reference
                        doc1_name = "Original Document"
                        doc2_name = compare_file.name
                        
                        # Extract text from comparison document
                        compare_text = extract_text_from_pdf(compare_file)
                        
                        # Generate comparison
                        comparison = compare_documents(st.session_state.full_text, compare_text)
                        
                        # Display comparison results
                        st.markdown(
                            f'<div style="border:1px solid #eee; padding:20px; border-radius:8px">'
                            f'{comparison}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Add export option
                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            if st.button("📊 Generate Detailed Comparison Report", use_container_width=True):
                                with st.spinner("Generating comparison report..."):
                                    report_buffer = export_comparison_report(
                                        st.session_state.full_text, 
                                        compare_text,
                                        doc1_name,
                                        doc2_name
                                    )
                                    st.session_state.comparison_report = report_buffer
                                    st.success("Comparison report ready for download!")
                        
                        with export_col2:
                            if st.session_state.get('comparison_report'):
                                st.download_button(
                                    label="⬇️ Download Comparison Report",
                                    data=st.session_state.comparison_report,
                                    file_name="Document_Comparison_Report.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")
                else:
                    st.info("Upload a second document to compare with your original document.")
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                        <h4 style="margin-top: 0;">Enhanced Document Comparison Features</h4>
                        <ul>
                            <li><strong>Section-by-Section Analysis</strong> - Compare specific parts of legal documents</li>
                            <li><strong>Semantic Comparison</strong> - Understand meaning differences, not just text</li>
                            <li><strong>Visual Highlighting</strong> - Clearly see additions, removals, and changes</li>
                            <li><strong>Statistical Analysis</strong> - Get metrics on document similarities and differences</li>
                            <li><strong>Exportable Reports</strong> - Generate standalone comparison reports</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze a document first before comparing.")

    # Compliance Section (Now as tab5)
    with tab5:
        st.header("Document Compliance Requirements")
        if st.session_state.document_processed and hasattr(st.session_state, 'document_compliance'):
            compliance_data = st.session_state.document_compliance
            
            if not compliance_data:
                st.info("No specific compliance requirements identified for this document.")
            else:
                for category, data in compliance_data.items():
                    confidence = data.get('confidence', 0)
                    with st.expander(f"📋 {category} Compliance (Confidence: {confidence})"):
                        # Requirements section
                        st.subheader("Key Compliance Requirements")
                        for item in data.get('requirements', []):
                            st.markdown(f'<div class="compliance-item">{item}</div>', unsafe_allow_html=True)
                        
                        # Relevant regulations section
                        st.subheader("Relevant Regulations")
                        regulations_html = ""
                        for regulation in data.get('relevant_regulations', []):
                            regulations_html += f'<span class="regulation-item">{regulation}</span>'
                        st.markdown(regulations_html, unsafe_allow_html=True)
                        
                        # Recent updates section
                        updates = data.get('updates', [])
                        if updates:
                            st.subheader("Recent Regulatory Updates")
                            for update in updates:
                                st.markdown(f"""
                                <div class="update-card">
                                    <div class="update-title">{update.get('title', '')}</div>
                                    <div class="update-source">Source: {update.get('source', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze a document to see relevant compliance requirements.")
            
            # Show sample compliance information
            with st.expander("📋 Sample Compliance Requirements"):
                st.subheader("Key Compliance Requirements")
                st.markdown("""
                <div class="compliance-item">📋 All parties properly identified and defined</div>
                <div class="compliance-item">🔍 Scope of work/services clearly outlined</div>
                <div class="compliance-item">💰 Payment terms and conditions specified</div>
                """, unsafe_allow_html=True)
                
                st.subheader("Relevant Regulations")
                st.markdown("""
                <span class="regulation-item">Uniform Commercial Code (UCC)</span>
                <span class="regulation-item">State Contract Laws</span>
                """, unsafe_allow_html=True)
    
    # Legal Updates Section (Now as tab6)
    with tab6:
        st.header("Document-Specific Legal Updates")
        if st.session_state.document_processed and hasattr(st.session_state, 'legal_updates'):
            legal_updates = st.session_state.legal_updates
            
            if not legal_updates:
                st.info("No relevant legal updates found for this document.")
            else:
                for category, data in legal_updates.items():
                    with st.expander(f"📋 {category} Updates (Confidence: {data['confidence']})"):
                        if not data['updates']:
                            st.info(f"No recent updates found for {category}")
                        else:
                            for update in data['updates']:
                                st.markdown(f"""
                                <div class="update-card">
                                    <div class="update-title">{update['title']}</div>
                                    <div class="update-source">Source: {update['source']}</div>
                                    <div class="update-snippet">{update.get('snippet', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze a document to see relevant legal updates.")
            
            # Demo button to show sample updates
            if st.button("Show Sample Updates"):
                st.markdown("""
                <div class="update-card">
                    <div class="update-title">GDPR Update: New Guidelines on Consent Management</div>
                    <div class="update-source">Source: https://gdpr-info.eu/</div>
                    <div class="update-snippet">The European Data Protection Board has published new guidelines on consent management practices, emphasizing the need for clear and affirmative consent collection methods...</div>
                </div>
                
                <div class="update-card">
                    <div class="update-title">Contract Law: Recent Court Decision on Force Majeure Clauses</div>
                    <div class="update-source">Source: https://www.americanbar.org/</div>
                    <div class="update-snippet">A recent Supreme Court decision has clarified the interpretation of force majeure clauses in commercial contracts, particularly in relation to pandemic-related business disruptions...</div>
                </div>
                """, unsafe_allow_html=True)

    # Report Section (Now as tab7)
    with tab7:
        st.header("Report Generation")
        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Generate PDF Report")
                
                if st.button("📊 Generate Full Report PDF", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        pdf_buffer = generate_pdf(
                            st.session_state.summaries['document'],
                            st.session_state.risk_data,
                            st.session_state.legal_updates if hasattr(st.session_state, 'legal_updates') else None,
                            st.session_state.document_compliance if hasattr(st.session_state, 'document_compliance') else None
                        )
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("PDF ready for download!")
                
                if st.session_state.get('pdf_buffer'):  # Check if pdf_buffer exists
                    st.download_button(
                        label="⬇️ Download Full Report",
                        data=st.session_state.pdf_buffer,
                        file_name="Legal_Analysis_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            # Email Reports Section
            with st.container(border=True):
                st.subheader("Email Reports")
                email = st.text_input("Recipient Email Address", placeholder="legal@company.com")
                
                email_col1, email_col2 = st.columns(2)
                
                with email_col1:
                    if st.button("📧 Send Summary Report", use_container_width=True):
                        if not email:
                            st.warning("Please enter an email address.")
                        else:
                            with st.spinner("Sending summary report..."):
                                # Create a summary-only PDF
                                summary_pdf = generate_pdf(
                                    st.session_state.summaries['document'],
                                    None,  # No risk data
                                    None,  # No legal updates
                                    None   # No compliance data
                                )
                                
                                # Create email content
                                email_html = create_email_text(
                                    summary=st.session_state.summaries['document']
                                )
                                
                                # Send email
                                success, message = send_email(
                                    email,
                                    summary_pdf,
                                    "Your Legal Document Summary Report",
                                    email_html,
                                    "document_summary.pdf"
                                )
                                
                                if success:
                                    st.success("Summary report sent successfully!")
                                else:
                                    st.error(message)
                
                with email_col2:
                    if st.button("📧 Send Complete Analysis", use_container_width=True):
                        if not email:
                            st.warning("Please enter an email address.")
                        elif not st.session_state.get('pdf_buffer'):
                            st.warning("Please generate the full report first.")
                        else:
                            with st.spinner("Sending complete analysis..."):
                                # Create email content with all sections
                                email_html = create_email_text(
                                    summary=st.session_state.summaries['document'],
                                    risk_assessment="Included in the attached PDF"
                                )
                                
                                # Send email with the full report
                                success, message = send_email(
                                    email,
                                    st.session_state.pdf_buffer,
                                    "Your Complete Legal Document Analysis",
                                    email_html,
                                    "complete_legal_analysis.pdf"
                                )
                                
                                if success:
                                    st.success("Complete analysis sent successfully!")
                                else:
                                    st.error(message)
        else:
            st.info("Please upload and analyze a document first to generate reports.")

if __name__ == "__main__":
    main()
