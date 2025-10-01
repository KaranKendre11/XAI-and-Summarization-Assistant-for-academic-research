import streamlit as st
import PyPDF2
import io
from transformers import pipeline
import time
import re

# Configure page
st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for jet black theme and animations
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
    
    .typing-animation {
        overflow: hidden;
        border-right: 0.15em solid #00d4ff;
        white-space: nowrap;
        animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
        font-family: monospace;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: #00d4ff; }
    }
    
    .stFileUploader > div > div {
        background-color: #1a1a1a;
        border: 2px dashed #00d4ff;
        border-radius: 10px;
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
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_summarizer():
    """Load and cache the summarization model with better error handling"""

    st.info("Attempting to load summarization model...")
    
    # Try BART first - it's more stable and reliable
    model = pipeline(
        "summarization", 
        "facebook/bart-large-cnn",
        device=0  
    )
    st.success("✅ PEGASUS model loaded successfully!")
    return model


# Initialize the model
if 'summarizer' not in st.session_state:
    with st.spinner("🤖 Loading AI model... This may take a moment on first run."):
        st.session_state.summarizer = load_summarizer()

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # Clean up the text
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'\s+', ' ', text)   # Remove multiple spaces
        text = text.strip()
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def summarize_text(text, max_length=150, min_length=50):
    """Summarize the extracted text with robust chunking and error handling"""
    try:
        # Clean and validate text
        text = text.strip()
        if not text:
            return "No text found to summarize."
        
        words = text.split()
        word_count = len(words)
        
        st.info(f"📊 Text statistics: {word_count} words, {len(text)} characters")
        
        # Conservative chunking approach - BART has max 1024 tokens
        max_words_per_chunk = 400  # Conservative limit
        
        # If text is short enough, process directly
        if word_count <= max_words_per_chunk:
            st.info("✅ Text is short enough for direct processing")
            
            # Adjust parameters based on text length
            adjusted_max = min(max_length, word_count // 3)
            adjusted_min = min(min_length, word_count // 6)
            adjusted_min = min(adjusted_min, adjusted_max - 10)  # Ensure min < max
            
            if adjusted_max < 20:
                adjusted_max = 20
            if adjusted_min < 5:
                adjusted_min = 5
            
            try:
                result = st.session_state.summarizer(
                    text,
                    max_length=adjusted_max,
                    min_length=adjusted_min,
                    do_sample=False,
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )
                return result[0]['summary_text']
            except Exception as direct_error:
                st.warning(f"Direct processing failed: {str(direct_error)}")
                # Return first few sentences as fallback
                sentences = text.split('.')[:5]
                return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
        
        # For longer texts, use careful chunking
        st.info(f"📝 Text is long ({word_count} words). Using smart chunking...")
        
        # Split into chunks by sentences, respecting word limits
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk_words = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            
            # If adding this sentence would exceed limit, save current chunk
            if current_word_count + len(sentence_words) > max_words_per_chunk and current_chunk_words:
                chunk_text = ' '.join(current_chunk_words) + '.'
                chunks.append(chunk_text)
                current_chunk_words = sentence_words
                current_word_count = len(sentence_words)
            else:
                current_chunk_words.extend(sentence_words)
                current_word_count += len(sentence_words)
        
        # Add remaining words as final chunk
        if current_chunk_words:
            chunk_text = ' '.join(current_chunk_words) + '.'
            chunks.append(chunk_text)
        
        st.info(f"📦 Created {len(chunks)} chunks for processing")
        
        # Process each chunk
        summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            progress_bar.progress((i + 1) / len(chunks))
            
            chunk_words = len(chunk.split())
            if chunk_words < 20:  # Skip very short chunks
                st.write(f"⏭️ Skipping short chunk {i+1}")
                continue
            
            st.write(f"🔄 Processing chunk {i+1}/{len(chunks)} ({chunk_words} words)")
            
            # Calculate chunk-specific parameters
            chunk_max = min(max_length // len(chunks) + 30, chunk_words // 3)
            chunk_min = min(min_length // len(chunks) + 10, chunk_words // 5)
            
            # Ensure valid parameters
            chunk_max = max(chunk_max, 15)
            chunk_min = max(chunk_min, 5)
            chunk_min = min(chunk_min, chunk_max - 5)
            
            try:
                result = st.session_state.summarizer(
                    chunk,
                    max_length=chunk_max,
                    min_length=chunk_min,
                    do_sample=False,
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )
                summary = result[0]['summary_text']
                summaries.append(summary)
                st.write(f"✅ Chunk {i+1} processed successfully")
                
            except Exception as chunk_error:
                st.warning(f"⚠️ Error with chunk {i+1}: {str(chunk_error)}")
                # Use first sentence of chunk as fallback
                fallback = chunk.split('.')[0] + '.'
                summaries.append(fallback)
                continue
        
        progress_bar.progress(1.0)
        
        if not summaries:
            return "❌ Unable to process any part of the document successfully."
        
        # Combine all summaries
        combined_summary = ' '.join(summaries)
        combined_words = len(combined_summary.split())
        
        st.success(f"✅ Generated combined summary ({combined_words} words)")
        
        # If combined summary is too long, do final summarization
        if combined_words > max_length * 1.5:
            st.info("🔄 Performing final consolidation...")
            
            try:
                final_max = min(max_length, combined_words // 2)
                final_min = min(min_length, combined_words // 4)
                final_min = min(final_min, final_max - 10)
                
                final_result = st.session_state.summarizer(
                    combined_summary,
                    max_length=final_max,
                    min_length=final_min,
                    do_sample=False,
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )
                return final_result[0]['summary_text']
                
            except Exception as final_error:
                st.warning(f"Final consolidation failed: {str(final_error)}")
                # Return truncated combined summary
                words = combined_summary.split()[:max_length]
                return ' '.join(words)
        
        return combined_summary
        
    except Exception as e:
        error_msg = f"Summarization error: {str(e)}"
        st.error(error_msg)
        
        # Last resort fallback
        try:
            st.info("🆘 Using emergency fallback...")
            sentences = text.split('.')[:8]  # Get first 8 sentences
            fallback_summary = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
            return f"[Fallback Summary] {fallback_summary}"
        except:
            return "❌ Unable to process this document. Please try a different PDF."

def animate_text_display(text):
    """Display text with typing animation effect"""
    import html
    # Escape HTML characters to prevent rendering issues
    safe_text = html.escape(text)
    
    placeholder = st.empty()
    
    # Display text character by character for animation effect
    displayed_text = ""
    for i, char in enumerate(safe_text):
        displayed_text += char
        # Use st.write instead of markdown for safer rendering
        placeholder.write(displayed_text)
        time.sleep(0.01)  # Reduced time for smoother animation
    

# Main UI
st.title("🤖 AI Research Paper Summarizer")
st.subheader("Upload a research paper PDF and get an intelligent summary")

# Chat-like interface
st.markdown("### 💬 Conversation")

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
    help="Upload a research paper PDF to get an AI-powered summary"
)

# Summarization parameters
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Maximum summary length", 150, 400, 150)
with col2:
    min_length = st.slider("Minimum summary length", 50, 300, 150)

if uploaded_file is not None:
    if st.button("🚀 Generate Summary", type="primary"):
        # Add user message
        st.session_state.messages.append({
            "role": "You", 
            "content": f"Uploaded: {uploaded_file.name}"
        })
        
        with st.spinner("🔍 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            st.success(f"✅ Extracted {len(text)} characters from PDF")
            
            # Show a preview of extracted text
            with st.expander("📖 Preview of extracted text"):
                st.text_area("Extracted content:", text[:1000] + "..." if len(text) > 1000 else text, height=200)
            
            with st.spinner("🤖 AI is generating summary..."):
                summary = summarize_text(text, max_length, min_length)
            
            if summary:
                # Add AI response to chat
                st.session_state.messages.append({
                    "role": "AI Assistant", 
                    "content": summary
                })
                
                # Display the summary with animation in a special container
                st.markdown("### ✨ AI-Generated Summary")
                with st.container():
                    animate_text_display(summary)
                
                # Provide summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{len(text):,} chars")
                with col2:
                    st.metric("Summary Length", f"{len(summary):,} chars")
                with col3:
                    compression_ratio = len(summary) / len(text) * 100
                    st.metric("Compression Ratio", f"{compression_ratio:.1f}%")

# Add some helpful information
with st.expander("ℹ️ How it works"):
    st.markdown("""
    This app uses Google's **BigBird-Pegasus** model specifically trained for summarizing academic papers:
    
    1. **Upload**: Select your research paper PDF
    2. **Extract**: The app extracts all text content from the PDF  
    3. **Process**: BigBird handles long documents efficiently using sparse attention
    4. **Summarize**: Get a concise, intelligent summary maintaining key insights
    5. **Interactive**: Chat-like interface for a conversational experience
    
    **Tips for best results:**
    - Use well-formatted academic PDFs
    - Papers with clear structure work best
    - Adjust summary length based on paper complexity
    """)

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit & Transformers | Powered by BigBird-Pegasus")