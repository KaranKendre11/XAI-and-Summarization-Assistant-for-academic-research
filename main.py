import streamlit as st
from model_handler import load_summarizer, summarize_text
from explainability_handler import BARTExplainabilityHandler
from simple_section_extractor import SimpleSectionExtractor

st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.35);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .feature-badge {
        animation: float 3s ease-in-out infinite;
    }
    
    .stMarkdown, .stText, p, span, div, label {
        color: #1a1a1a !important;
    }
    
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: scale(1.01);
    }
    
    .stFileUploader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed rgba(102, 126, 234, 0.4);
        transition: all 0.4s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        box-shadow: 0 8px 40px rgba(102, 126, 234, 0.3);
        transform: translateY(-3px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        color: #667eea !important;
        border: 2px solid rgba(102, 126, 234, 0.4);
        border-radius: 15px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: white !important;
        border-color: #667eea;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        color: #667eea !important;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    .stCheckbox {
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        transform: translateX(3px);
    }
    
    .stSuccess, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        animation: slideInUp 0.5s ease-out;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: scale(1.05);
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: bold;
    }
    
    .stSlider {
        padding: 1rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #667eea;
        background-color: transparent;
        border-radius: 8px;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stSpinner > div {
        display: none !important;
    }
    
    .stSpinner {
        text-align: center;
        color: white !important;
        font-weight: 600;
    }
    
    .stSpinner::after {
        content: '⏳ Processing...';
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

for key in ['summary', 'original_text', 'detected_sections', 'selected_sections', 'summarized_text']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'modal_reset' not in st.session_state:
    st.session_state.modal_reset = True
    
if st.session_state.modal_reset:
    st.session_state.show_summary_modal = False
    st.session_state.show_explain_modal = False

if 'summarizer' not in st.session_state:
    with st.spinner("Loading AI model..."):
        summarizer_result = load_summarizer()
        if isinstance(summarizer_result, tuple):
            st.session_state.summarizer, st.session_state.tokenizer = summarizer_result
        else:
            st.session_state.summarizer = summarizer_result
            from transformers import AutoTokenizer
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        st.session_state.explainability_handler = BARTExplainabilityHandler(
            st.session_state.summarizer,
            st.session_state.tokenizer
        )
        st.session_state.section_extractor = SimpleSectionExtractor()

st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin: 1rem 0;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    animation: titleFloat 3s ease-in-out infinite;
">
    🔬 AI Research Paper Summarizer
</h1>
<style>
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
    text-align: center;
    color: rgba(255, 255, 255, 0.95);
    font-size: 1rem;
    margin-bottom: 1.5rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
">
    Upload a research paper PDF and get an intelligent summary
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1.5rem 0;
">
    <div class="feature-badge" style="
        animation-delay: 0s;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    ">
        <p style="color: #667eea; font-weight: 600; margin: 0; font-size: 0.95rem;">🚀 BART Model</p>
    </div>
    <div class="feature-badge" style="
        animation-delay: 0.2s;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    ">
        <p style="color: #667eea; font-weight: 600; margin: 0; font-size: 0.95rem;">📊 Sections</p>
    </div>
    <div class="feature-badge" style="
        animation-delay: 0.4s;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    ">
        <p style="color: #667eea; font-weight: 600; margin: 0; font-size: 0.95rem;">🎯 XAI</p>
    </div>
    <div class="feature-badge" style="
        animation-delay: 0.6s;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    ">
        <p style="color: #667eea; font-weight: 600; margin: 0; font-size: 0.95rem;">⚡ GPU</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: none; height: 1px; background: rgba(255, 255, 255, 0.3); margin: 1.5rem 0;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file and not st.session_state.detected_sections:
    with st.spinner("Processing..."):
        sections = st.session_state.section_extractor.extract_sections(uploaded_file)
        st.session_state.original_text = sections.get("Full Paper", "")
        st.session_state.detected_sections = sections
        st.session_state.selected_sections = ["Full Paper"]
        st.rerun()

if st.session_state.detected_sections:
    available = [k for k in st.session_state.detected_sections.keys() if k != "Full Paper"]
    if available:
        st.success(f"✓ Found {len(available)} sections")

if st.session_state.detected_sections:
    with st.expander("📑 Select Sections"):
        sections = st.session_state.detected_sections
        
        section_order = [
            'Full Paper', 'Abstract', 'Introduction', 'Background', 'Related Work',
            'Literature Review', 'Methodology', 'Methods', 'Approach',
            'Materials and Methods', 'Experimental Setup', 'Experiments',
            'Results', 'Results and Discussion', 'Discussion', 'Analysis',
            'Evaluation', 'Conclusion', 'Conclusions', 'Future Work',
            'Acknowledgments', 'Acknowledgements', 'References'
        ]
        
        names = [s for s in section_order if s in sections]
        for s in sections.keys():
            if s not in names:
                names.append(s)
        
        st.info("💡 'Full Paper' includes all sections")
        
        select_all = st.checkbox("Select Full Paper", value=True, key="sel_all")
        
        if select_all:
            st.session_state.selected_sections = ["Full Paper"]
        else:
            selected = []
            cols = st.columns(3)
            col_idx = 0
            
            for i, name in enumerate(names):
                if name == "Full Paper":
                    continue
                    
                with cols[col_idx % 3]:
                    if st.checkbox(name, value=False, key=f"sec_{i}"):
                        selected.append(name)
                col_idx += 1
            
            st.session_state.selected_sections = selected if selected else ["Full Paper"]

if st.session_state.detected_sections:
    with st.expander("⚙️ Settings"):
        col1, col2 = st.columns(2)
        max_length = col1.slider("Max length", 50, 400, 150)
        min_length = col2.slider("Min length", 30, 200, 50)
    
    st.markdown("---")
    
    if st.button("🚀 Generate Summary", type="primary"):
        if "Full Paper" in st.session_state.selected_sections:
            text = st.session_state.detected_sections["Full Paper"]
        else:
            parts = [st.session_state.detected_sections[s] for s in st.session_state.selected_sections if s in st.session_state.detected_sections]
            text = '\n\n'.join(parts)
        
        with st.spinner("Generating..."):
            summary = summarize_text(
                st.session_state.summarizer,
                st.session_state.tokenizer,
                text,
                max_length,
                min_length
            )
            st.session_state.summary = summary
            st.session_state.summarized_text = text
        
        st.success("✓ Done!")
        st.rerun()

if st.session_state.summary:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 View Summary", key="btn_summary"):
            st.session_state.show_summary_modal = True
            st.session_state.show_explain_modal = False
            st.session_state.modal_reset = False
            st.rerun()
    
    with col2:
        if st.button("🔬 Explainability", key="btn_explain"):
            st.session_state.show_explain_modal = True
            st.session_state.show_summary_modal = False
            st.session_state.modal_reset = False
            st.rerun()
    
    with col3:
        st.download_button("💾 Download", data=st.session_state.summary, file_name="summary.txt", key="btn_download")

if st.session_state.get('show_summary_modal', False):
    @st.dialog("📄 Summary", width="large")
    def summary_modal():
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; line-height: 1.8;">
            <p style="color: #1a1a1a; font-size: 1.05rem;">{st.session_state.summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        text_len = len(st.session_state.summarized_text)
        sum_len = len(st.session_state.summary)
        col1.metric("Original", f"{text_len:,}")
        col2.metric("Summary", f"{sum_len:,}")
        col3.metric("Compression", f"{(1-sum_len/text_len)*100:.0f}%")
    
    summary_modal()
    st.session_state.modal_reset = True

if st.session_state.get('show_explain_modal', False):
    @st.dialog("🔬 Explainability Analysis", width="large")
    def explain_modal():
        st.info("Analyzing how the model created the summary...")
        
        st.session_state.explainability_handler.display_explainability_dashboard(
            st.session_state.summarized_text,
            st.session_state.summary
        )
    
    explain_modal()
    st.session_state.modal_reset = True