import streamlit as st
import PyPDF2
import io
import time
import re
import html
from model_handler import load_summarizer, summarize_text
from explainability_handler import BARTExplainabilityHandler  # New import

# Configure page
st.set_page_config(
    page_title="AI Research Paper Summarizer with Explainability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for jet black theme with explainability features
st.markdown("""
<style>
    .stApp {
        background-color: #0e0e0e;
        color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        color: #00d4ff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
        50% { text-shadow: 0 0 30px rgba(0, 212, 255, 0.8); }
        100% { text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
    }
    
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .summary-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3); }
        to { box-shadow: 0 15px 40px rgba(0, 212, 255, 0.6); }
    }
    
    .explainability-header {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
    }
    
    .stFileUploader > div > div {
        background-color: #1a1a1a;
        border: 2px dashed #00d4ff;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
        background: linear-gradient(45deg, #00ffff, #00aadd);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888888;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #000000 !important;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'summary' not in st.session_state:
    st.session_state.summary = None

if 'original_text' not in st.session_state:
    st.session_state.original_text = None

if 'show_explainability' not in st.session_state:
    st.session_state.show_explainability = False

# Initialize the model and explainability handler
if 'summarizer' not in st.session_state:
    with st.spinner("🤖 Loading AI model... This may take a moment on first run."):
        summarizer_result = load_summarizer()
        if isinstance(summarizer_result, tuple):
            st.session_state.summarizer, st.session_state.tokenizer = summarizer_result
        else:
            st.session_state.summarizer = summarizer_result
            from transformers import AutoTokenizer
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Initialize explainability handler
        st.session_state.explainability_handler = BARTExplainabilityHandler(
            st.session_state.summarizer,
            st.session_state.tokenizer
        )

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # Clean up the text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def animate_text_display(text):
    """Display text with typing animation effect"""
    safe_text = html.escape(text)
    placeholder = st.empty()
    
    displayed_text = ""
    for i, char in enumerate(safe_text):
        displayed_text += char
        placeholder.write(displayed_text)
        time.sleep(0.005)  # Faster animation

# Main UI
st.markdown('<h1 class="main-header">🔬 AI Research Paper Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a research paper PDF and get an intelligent summary with explainability insights</p>', unsafe_allow_html=True)

# Add feature badges
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🚀 **BART Model**")
with col2:
    st.markdown("📊 **Explainable AI**")
with col3:
    st.markdown("🎯 **Attention Viz**")
with col4:
    st.markdown("⚡ **GPU Accelerated**")

st.markdown("---")

# Chat-like interface
st.markdown("### 💬 Conversation History")

# Display chat history
for message in st.session_state.messages:
    with st.container():
        role = message["role"]
        content = message["content"]
        st.write(f"**{role}:** {content}")

# File upload section
st.markdown("### 📄 Upload Research Paper")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a research paper PDF to get an AI-powered summary with explainability"
)

# Summarization parameters
st.markdown("### ⚙️ Summarization Settings")
col1, col2, col3 = st.columns(3)
with col1:
    max_length = st.slider("Maximum summary length", 50, 400, 150)
with col2:
    min_length = st.slider("Minimum summary length", 30, 200, 50)
with col3:
    enable_explainability = st.checkbox("🔬 Enable Explainability", value=True, 
                                       help="Show detailed explanations of how the AI created the summary")

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            # Add user message
            st.session_state.messages.append({
                "role": "You", 
                "content": f"Uploaded: {uploaded_file.name}"
            })
            
            with st.spinner("🔍 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.success(f"✅ Extracted {len(text)} characters from PDF")
                st.session_state.original_text = text
                
                # Show a preview of extracted text
                with st.expander("📖 Preview of extracted text"):
                    st.text_area("Extracted content:", 
                               text[:1000] + "..." if len(text) > 1000 else text, 
                               height=200)

                with st.spinner("🤖 AI is generating summary... This may take a moment."):
                    summary = summarize_text(
                        st.session_state.summarizer, 
                        st.session_state.tokenizer,
                        text, 
                        max_length, 
                        min_length
                    )
                    st.session_state.summary = summary
                
                if summary:
                    # Add AI response to chat
                    st.session_state.messages.append({
                        "role": "AI Assistant", 
                        "content": summary
                    })
                    
                    # Display the summary with animation in a special container
                    st.markdown("### ✨ AI-Generated Summary")
                    st.markdown('<div class="summary-container">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Provide summary statistics
                    st.markdown("### 📊 Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Length", f"{len(text):,} chars")
                    with col2:
                        st.metric("Summary Length", f"{len(summary):,} chars")
                    with col3:
                        compression_ratio = (1 - len(summary) / len(text)) * 100
                        st.metric("Compression", f"{compression_ratio:.1f}%")
                    
                    # Set flag to show explainability
                    if enable_explainability:
                        st.session_state.show_explainability = True
    
    with col2:
        if st.session_state.summary and st.session_state.original_text:
            if st.button("🔬 Analyze Explainability", type="secondary", use_container_width=True):
                st.session_state.show_explainability = True

# Show explainability dashboard if enabled
if st.session_state.show_explainability and st.session_state.summary and st.session_state.original_text:
    st.markdown("---")
    st.markdown('<h2 class="explainability-header">🔬 Explainability Analysis</h2>', unsafe_allow_html=True)
    
    # Display the explainability dashboard
    st.session_state.explainability_handler.display_explainability_dashboard(
        st.session_state.original_text,
        st.session_state.summary
    )
    
    # Add download options for explainability report
    st.markdown("### 💾 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Summary", use_container_width=True):
            # Create downloadable text file
            summary_text = f"""
AI-GENERATED SUMMARY
==================
{st.session_state.summary}

STATISTICS
==========
Original Length: {len(st.session_state.original_text)} chars
Summary Length: {len(st.session_state.summary)} chars
Compression: {(1 - len(st.session_state.summary) / len(st.session_state.original_text)) * 100:.1f}%
            """
            st.download_button(
                label="Download Summary (.txt)",
                data=summary_text,
                file_name="summary_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔄 Reset & Start New", use_container_width=True):
            st.session_state.summary = None
            st.session_state.original_text = None
            st.session_state.show_explainability = False
            st.rerun()

# Add helpful information
with st.sidebar:
    st.markdown("## ℹ️ How it works")
    st.markdown("""
    This advanced summarizer uses:
    
    **🤖 BART Model**
    - State-of-the-art transformer
    - Pre-trained on CNN/DailyMail
    - Generates human-like summaries
    
    **🔬 Explainability Features**
    - **Attention Visualization**: See what the model "looks at"
    - **Sentence Importance**: Which sentences matter most
    - **Token Attribution**: Word-level importance scores
    - **Summary Statistics**: Extraction vs abstraction metrics
    
    **📊 Visualization Types**
    1. **Attention Heatmaps**: Model's focus patterns
    2. **Importance Scores**: Content contribution analysis
    3. **Token Highlighting**: Visual importance mapping
    4. **Statistical Analysis**: Compression and novelty metrics
    
    **Tips for best results:**
    - Use well-formatted PDFs
    - Longer documents work better
    - Adjust summary length as needed
    - Enable explainability for insights
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Understanding the Visualizations")
    
    with st.expander("Attention Heatmap"):
        st.markdown("""
        Shows which input words the model "looked at" when generating each word of the summary.
        - Brighter = higher attention
        - Helps understand model focus
        """)
    
    with st.expander("Sentence Importance"):
        st.markdown("""
        Ranks sentences by their contribution to the final summary.
        - Based on semantic similarity
        - Identifies key content
        """)
    
    with st.expander("Token Highlighting"):
        st.markdown("""
        Color-codes input text by importance:
        - 🔴 Red = Critical tokens
        - 🟠 Orange = Important tokens
        - 🟡 Yellow = Relevant tokens
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>Built with ❤️ using Streamlit, Transformers & Custom Explainability</p>
    <p style="font-size: 0.9em;">Powered by BART • Attention Visualization • Token Attribution</p>
</div>
""", unsafe_allow_html=True)