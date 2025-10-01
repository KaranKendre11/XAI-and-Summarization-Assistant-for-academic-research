from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import streamlit as st
import nltk
import re
import torch
import os

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@st.cache_resource
def load_summarizer():
    """Load BART with explicit model control to avoid CUDA errors"""
    model_name = "facebook/bart-large-cnn"
    
    # Load tokenizer first
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load model with specific settings
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for stability
        low_cpu_mem_usage=True
    )
    
    # Move to GPU after loading
    if torch.cuda.is_available():
        model = model.cuda()
        st.success("✅ Model loaded on GPU!")
    else:
        st.success("✅ Model loaded on CPU!")
    
    # Create custom pipeline wrapper
    class SimpleSummarizer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            
        def __call__(self, text, max_length=150, min_length=50, **kwargs):
            # Clean text to ASCII only
            text = text.encode('ascii', 'ignore').decode('ascii')
            
            # Tokenize with truncation
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return [{"summary_text": summary}]
    
    summarizer = SimpleSummarizer(model, tokenizer)
    
    return summarizer, tokenizer

def fix_tokenization_issues(text, tokenizer):
    """Fix text to prevent tokenization errors"""
    # Remove all characters that could cause issues
    # Keep only ASCII printable characters
    cleaned = ''.join(char for char in text if 32 <= ord(char) < 127)
    
    # Test tokenization to ensure it works
    try:
        # Try to tokenize - if it fails, clean more aggressively
        test_tokens = tokenizer.encode(cleaned, add_special_tokens=True, truncation=True, max_length=512)
        tokenizer.decode(test_tokens)  # Test decoding too
        return cleaned
    except:
        # If still failing, use only basic ASCII letters and numbers
        ultra_clean = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', cleaned)
        return ultra_clean

def safe_tokenize_and_summarize(summarizer, text, max_length=150, min_length=50):
    """Safely tokenize and summarize with CUDA error prevention"""
    try:
        # First, ensure text is within vocabulary
        tokenizer = summarizer.tokenizer
        
        # Clean text to prevent token index errors
        clean_text = fix_tokenization_issues(text, tokenizer)
        
        # Tokenize with explicit truncation
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=False
        )
        
        # Check for invalid token indices
        max_token_id = tokenizer.vocab_size - 1
        if torch.any(inputs.input_ids > max_token_id):
            # Remove any tokens outside vocabulary
            inputs.input_ids[inputs.input_ids > max_token_id] = tokenizer.pad_token_id
        
        # Move to device if using GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate summary with error handling
        with torch.no_grad():
            summary_ids = summarizer.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            st.error(f"CUDA error: {e}")
            # Clear CUDA cache and try CPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            raise
        else:
            raise

def clean_text_for_bart(text):
    """Clean text specifically for BART tokenizer"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    # Remove Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove citations
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    
    # Clean special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def chunk_text_safely(text, tokenizer, max_tokens=900):
    """Chunk text with vocabulary safety checks"""
    chunks = []
    words = text.split()
    current_chunk = []
    
    for word in words:
        # Test if adding this word would cause tokenization issues
        test_text = ' '.join(current_chunk + [word])
        try:
            tokens = tokenizer.encode(test_text, add_special_tokens=False, truncation=True)
            if len(tokens) > max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                current_chunk = [word]
            else:
                current_chunk.append(word)
        except:
            # Skip problematic words
            continue
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(summarizer, tokenizer, text, max_length=150, min_length=50):
    """Main summarization function with CUDA error prevention"""
    try:
        if not text:
            return "No text provided."
        
        # Clean text
        text = clean_text_for_bart(text)
        
        if len(text) < 100:
            return "Text too short to summarize."
        
        # Check if we can process directly
        try:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        except:
            # If tokenization fails, clean more aggressively
            text = fix_tokenization_issues(text, tokenizer)
            token_count = len(text.split()) * 2  # Estimate
        
        st.info(f"📊 Processing ~{token_count:,} tokens")
        
        # If text is short enough, process directly
        if token_count <= 900:
            st.info("✅ Processing directly...")
            try:
                # Use the safe summarization function
                summary = safe_tokenize_and_summarize(summarizer, text, max_length, min_length)
                return summary
            except Exception as e:
                st.error(f"Direct processing failed: {e}")
                return fallback_summary(text)
        
        # Otherwise, chunk the text
        st.info("📦 Chunking text...")
        chunks = chunk_text_safely(text, tokenizer, max_tokens=900)
        st.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            return "Unable to process text."
        
        # Process each chunk
        summaries = []
        progress = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            progress.progress((i + 1) / len(chunks))
            st.write(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                chunk_summary = safe_tokenize_and_summarize(
                    summarizer, 
                    chunk,
                    max_length=max(50, max_length // len(chunks)),
                    min_length=30
                )
                if chunk_summary:
                    summaries.append(chunk_summary)
            except Exception as e:
                st.warning(f"Chunk {i+1} failed: {e}")
                # Use fallback for this chunk
                summaries.append(chunk[:200] + "...")
        
        progress.progress(1.0)
        
        if not summaries:
            return "Failed to generate summary."
        
        # Combine summaries
        combined = ' '.join(summaries)
        
        # If combined is too long, summarize again
        if len(combined.split()) > max_length * 2:
            st.info("🔄 Consolidating final summary...")
            try:
                final = safe_tokenize_and_summarize(summarizer, combined, max_length, min_length)
                return final
            except:
                # Return truncated combined
                words = combined.split()[:max_length]
                return ' '.join(words)
        
        return combined
        
    except Exception as e:
        st.error(f"Error: {e}")
        # Clear CUDA state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return fallback_summary(text)

def fallback_summary(text):
    """Simple fallback when GPU fails"""
    words = text.split()[:150]
    return ' '.join(words) + "..."

# Main execution
if __name__ == "__main__":
    st.title("📄 CUDA-Safe BART Summarizer")
    
    # Show GPU status
    if torch.cuda.is_available():
        st.success(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        st.info(f"CUDA Version: {torch.version.cuda}")
    else:
        st.warning("⚠️ Running on CPU")
    
    # Load model
    summarizer, tokenizer = load_summarizer()
    
    # Text input
    text = st.text_area("Enter text to summarize:", height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max length", 50, 300, 150)
    with col2:
        min_len = st.slider("Min length", 20, 100, 50)
    
    if st.button("Summarize"):
        if text:
            with st.spinner("Generating summary..."):
                summary = summarize_text(summarizer, tokenizer, text, max_len, min_len)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.warning("Please enter text to summarize")