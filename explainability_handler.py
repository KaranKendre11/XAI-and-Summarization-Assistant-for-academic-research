import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import colorsys

class BARTExplainabilityHandler:
    """Explainability module for BART summarization with practical visualization methods"""
    
    def __init__(self, summarizer, tokenizer):
        """
        Initialize with either a model or a SimpleSummarizer wrapper
        """
        # Check if it's a SimpleSummarizer wrapper or direct model
        if hasattr(summarizer, 'model'):
            # It's a SimpleSummarizer wrapper
            self.model = summarizer.model
            self.tokenizer = summarizer.tokenizer if hasattr(summarizer, 'tokenizer') else tokenizer
        else:
            # It's a direct model
            self.model = summarizer
            self.tokenizer = tokenizer
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move model to device if it's not already there
        if self.device == 'cuda' and hasattr(self.model, 'cuda'):
            try:
                self.model = self.model.cuda()
            except:
                # If model is already on CUDA or doesn't support .cuda(), continue
                pass
    
    def calculate_sentence_importance(self, text: str, summary: str) -> List[Tuple[str, float]]:
        """
        Calculate importance scores for each sentence in the input.
        Shows which sentences contributed most to the summary.
        """
        # Split into sentences (simple approach)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Create TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            
            # Fit on all text
            all_text = sentences + [summary]
            tfidf_matrix = vectorizer.fit_transform(all_text)
            
            # Get similarity between each sentence and summary
            summary_vector = tfidf_matrix[-1]
            sentence_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(sentence_vectors, summary_vector).flatten()
            
            # Combine sentences with scores
            sentence_scores = list(zip(sentences, similarities))
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            return sentence_scores
        except Exception as e:
            # Fallback: return sentences with uniform scores
            return [(s, 1.0/len(sentences)) for s in sentences]
    
    def get_token_importance_scores(self, text: str, summary: str) -> Dict:
        """
        Calculate importance scores for input tokens based on TF-IDF.
        (Attention-based method disabled due to performance issues)
        """
        # Use TF-IDF scores directly
        tokens = self.tokenizer.tokenize(text[:10000])  # Limit text length
        
        # Simple TF-IDF based importance
        from collections import Counter
        
        # Count token frequency in text and summary
        text_counts = Counter(tokens)
        summary_tokens = self.tokenizer.tokenize(summary)
        summary_counts = Counter(summary_tokens)
        
        # Calculate importance scores
        scores = []
        for token in tokens:
            # Score based on presence in summary and frequency
            score = 0.3  # Base score
            if token in summary_counts:
                score += 0.5 * (summary_counts[token] / len(summary_tokens))
            if token in text_counts:
                score += 0.2 * (1 / text_counts[token])  # Inverse frequency
            scores.append(min(score, 1.0))
        
        return {
            'tokens': tokens,
            'scores': np.array(scores),
            'summary': summary
        }
    
    def create_sentence_importance_chart(self, sentence_scores: List[Tuple[str, float]]) -> go.Figure:
        """Create bar chart showing sentence importance"""
        
        # Take top 10 sentences
        top_sentences = sentence_scores[:10]
        
        # Truncate long sentences for display
        display_sentences = []
        for sent, score in top_sentences:
            if len(sent) > 80:
                display_sent = sent[:77] + "..."
            else:
                display_sent = sent
            display_sentences.append((display_sent, score))
        
        sentences = [s for s, _ in display_sentences]
        scores = [score for _, score in display_sentences]
        
        # Create gradient colors based on scores
        colors = [f'rgba(102, 126, 234, {0.5 + s*0.5})' for s in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=sentences,
                orientation='h',
                marker_color=colors,
                text=[f'{s:.3f}' for s in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Most Important Sentences for Summary",
            title_font=dict(size=18, color='#667eea', family='Segoe UI'),
            xaxis_title="Importance Score",
            yaxis_title="",
            height=400,
            template='plotly_white',
            showlegend=False,
            xaxis=dict(range=[0, max(scores) * 1.1] if scores else [0, 1]),
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(color='#1a1a1a')
        )
        
        return fig
    
    def create_token_highlighting(self, text: str, token_scores: Dict) -> str:
        """
        Create HTML with highlighted tokens based on importance scores.
        Returns HTML string for st.markdown with unsafe_allow_html=True.
        """
        if not token_scores:
            return text
        
        tokens = token_scores['tokens']
        scores = token_scores['scores']
        
        # Create HTML with color-coded tokens
        html_parts = []
        
        for token, score in zip(tokens, scores):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '<unk>']:
                continue
            
            # Convert score to color (red = high importance, light = low)
            if score > 0.7:
                color = f'background-color: rgba(255, 100, 100, 0.6); font-weight: bold;'
            elif score > 0.4:
                color = f'background-color: rgba(255, 180, 100, 0.5);'
            elif score > 0.2:
                color = f'background-color: rgba(255, 220, 100, 0.4);'
            else:
                color = ''
            
            # Clean token (remove special characters from different tokenizers)
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').replace('##', '')
            
            if color:
                html_parts.append(f'<span style="{color} padding: 2px 4px; border-radius: 4px; color: #1a1a1a;">{clean_token}</span>')
            else:
                html_parts.append(f'<span style="color: #1a1a1a;">{clean_token}</span>')
        
        html_text = ''.join(html_parts)
        
        # White glass container
        return f"""
        <div style="
            padding: 2rem; 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px; 
            border: 1px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            line-height: 2;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            word-wrap: break-word;
        ">
            <h4 style="color: #667eea; margin-bottom: 0.5rem; font-weight: 600;">
                Token Importance Visualization
            </h4>
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 1.5rem;">
                Highlighted colors show importance for generating the summary
            </p>
            <div style="color: #1a1a1a;">
                {html_text}
            </div>
        </div>
        """
    
    def generate_explainability_report(self, text: str, summary: str) -> Dict:
        """
        Generate comprehensive explainability report with all visualizations.
        """
        report = {}
        
        # Attention extraction disabled to prevent hanging
        st.info("Attention visualization disabled for performance. Using TF-IDF based visualizations.")
        report['attention_heatmap'] = None
        
        # Get sentence importance with error handling
        try:
            with st.spinner("Calculating sentence importance..."):
                sentence_scores = self.calculate_sentence_importance(text, summary)
                if sentence_scores:
                    report['sentence_chart'] = self.create_sentence_importance_chart(sentence_scores)
                    report['sentence_scores'] = sentence_scores
        except Exception as e:
            st.warning(f"Could not calculate sentence importance: {str(e)}")
        
        # Get token importance with error handling  
        try:
            with st.spinner("Computing token importance..."):
                token_scores = self.get_token_importance_scores(text, summary)
                if token_scores:
                    report['token_html'] = self.create_token_highlighting(text, token_scores)
                    report['token_scores'] = token_scores
        except Exception as e:
            st.warning(f"Could not compute token importance: {str(e)}")
        
        return report
    
    def display_explainability_dashboard(self, text: str, summary: str):
        """
        Display complete explainability dashboard in Streamlit.
        """
        # Generate report
        report = self.generate_explainability_report(text, summary)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Sentence Importance", "Token Importance", "Summary Statistics"])
        
        with tab1:
            if 'sentence_chart' in report:
                st.plotly_chart(report['sentence_chart'], use_container_width=True)
                
                with st.expander("View all sentence scores"):
                    for i, (sent, score) in enumerate(report['sentence_scores'], 1):
                        if score > 0.1:
                            st.write(f"**{i}. Score: {score:.3f}**")
                            st.write(f"{sent[:200]}..." if len(sent) > 200 else sent)
                            st.divider()
        
        with tab2:
            if 'token_html' in report:
                st.markdown(report['token_html'], unsafe_allow_html=True)
                
                # Add legend with glass effect
                st.markdown("""
                <div style="
                    margin-top: 1.5rem; 
                    padding: 1.5rem; 
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 12px;
                    border: 1px solid rgba(102, 126, 234, 0.2);
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                ">
                    <p style="color: #667eea; font-size: 1rem; font-weight: 600; margin-bottom: 0.8rem;">
                        Legend
                    </p>
                    <p style="color: #1a1a1a; font-size: 0.95rem; line-height: 1.6;">
                        <span style="background-color: rgba(255, 100, 100, 0.6); padding: 3px 8px; border-radius: 4px; font-weight: bold;">High</span> importance (>0.7) | 
                        <span style="background-color: rgba(255, 180, 100, 0.5); padding: 3px 8px; border-radius: 4px;">Medium</span> importance (0.4-0.7) | 
                        <span style="background-color: rgba(255, 220, 100, 0.4); padding: 3px 8px; border-radius: 4px;">Low</span> importance (0.2-0.4)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Calculate statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Length", f"{len(text.split())} words")
                st.metric("Summary Length", f"{len(summary.split())} words")
                compression = (1 - len(summary) / len(text)) * 100
                st.metric("Compression Rate", f"{compression:.1f}%")
            
            with col2:
                # Calculate extraction vs abstraction ratio
                original_words = set(text.lower().split())
                summary_words = set(summary.lower().split())
                common_words = original_words.intersection(summary_words)
                extraction_ratio = len(common_words) / len(summary_words) * 100 if summary_words else 0
                
                st.metric("Word Overlap", f"{extraction_ratio:.1f}%")
                st.metric("Novel Words", f"{100 - extraction_ratio:.1f}%")
                
                # Abstractiveness indicator
                if extraction_ratio > 70:
                    st.warning("Highly Extractive Summary")
                elif extraction_ratio > 40:
                    st.info("Balanced Extractive-Abstractive")
                else:
                    st.success("Highly Abstractive Summary")