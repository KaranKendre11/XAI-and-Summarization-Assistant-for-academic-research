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
    
    def get_attention_weights(self, text: str, summary: str = None, max_length: int = 150) -> Dict:
        """
        Extract attention weights from BART model during generation or for given summary.
        This shows which parts of the input the model focused on.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        if self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # If no summary provided, generate one
        if summary is None:
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True,
                        output_attentions=True,
                        return_dict_in_generate=True
                    )
                    summary_ids = outputs.sequences
                except:
                    # Fallback for models that don't support output_attentions in generate
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            # Use provided summary
            summary_inputs = self.tokenizer(summary, return_tensors="pt")
            if self.device == 'cuda':
                summary_inputs = {k: v.cuda() for k, v in summary_inputs.items()}
            summary_ids = summary_inputs["input_ids"]
        
        # Get model outputs with attention
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    decoder_input_ids=summary_ids,
                    output_attentions=True,
                    return_dict=True
                )
            
            # Extract cross-attention (decoder attending to encoder)
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                # Get last layer's cross attention
                cross_attention = outputs.cross_attentions[-1]  # Shape: [batch, heads, decoder_len, encoder_len]
                
                # Average across attention heads
                avg_attention = cross_attention.mean(dim=1).squeeze(0).cpu().numpy()
                
                # Get tokens
                input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
                summary_tokens = self.tokenizer.convert_ids_to_tokens(summary_ids[0].cpu())
                
                return {
                    'attention_matrix': avg_attention,
                    'input_tokens': input_tokens,
                    'summary_tokens': summary_tokens,
                    'summary': summary
                }
        except Exception as e:
            st.warning(f"Could not extract attention weights: {str(e)}. Using fallback visualization.")
            # Return None to trigger fallback visualizations
            return None
        
        return None
    
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
        Calculate importance scores for input tokens based on attention patterns.
        If attention weights are not available, use TF-IDF as fallback.
        """
        attention_data = self.get_attention_weights(text, summary)
        
        if attention_data:
            # Use attention weights
            token_importance = attention_data['attention_matrix'].sum(axis=0)
            
            # Normalize scores
            if token_importance.max() > 0:
                token_importance = token_importance / token_importance.max()
            
            return {
                'tokens': attention_data['input_tokens'],
                'scores': token_importance,
                'summary': summary
            }
        else:
            # Fallback: Use TF-IDF scores
            tokens = self.tokenizer.tokenize(text)
            
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
    
    def create_attention_heatmap(self, attention_data: Dict) -> go.Figure:
        """Create interactive attention heatmap using Plotly"""
        
        if not attention_data or 'attention_matrix' not in attention_data:
            # Return None if no attention data available
            return None
        
        attention_matrix = attention_data['attention_matrix']
        input_tokens = attention_data['input_tokens']
        summary_tokens = attention_data['summary_tokens']
        
        # Limit tokens for better visualization
        max_input = min(50, len(input_tokens))
        max_summary = min(30, len(summary_tokens))
        
        attention_matrix = attention_matrix[:max_summary, :max_input]
        input_tokens = input_tokens[:max_input]
        summary_tokens = summary_tokens[:max_summary]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=input_tokens,
            y=summary_tokens,
            colorscale='Viridis',
            text=np.round(attention_matrix, 3),
            hovertemplate='Input: %{x}<br>Summary: %{y}<br>Attention: %{text}<extra></extra>',
            colorbar=dict(title="Attention<br>Weight")
        ))
        
        fig.update_layout(
            title="Attention Visualization: How the model reads your text",
            xaxis_title="Input Text Tokens",
            yaxis_title="Generated Summary Tokens",
            height=500,
            xaxis={'tickangle': -45},
            template='plotly_dark',
            font=dict(size=10)
        )
        
        return fig
    
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
        colors = [f'rgba(0, {int(255*s)}, {int(255*(1-s))}, 0.8)' for s in scores]
        
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
            title="📊 Most Important Sentences for Summary",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=400,
            template='plotly_dark',
            showlegend=False,
            xaxis=dict(range=[0, max(scores) * 1.1] if scores else [0, 1])
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
                color = f'background-color: rgba(255, 0, 0, {score*0.5}); font-weight: bold;'
            elif score > 0.4:
                color = f'background-color: rgba(255, 165, 0, {score*0.4});'
            elif score > 0.2:
                color = f'background-color: rgba(255, 255, 0, {score*0.3});'
            else:
                color = ''
            
            # Clean token (remove special characters from different tokenizers)
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').replace('##', '')
            
            if color:
                html_parts.append(f'<span style="{color} padding: 2px; border-radius: 3px;">{clean_token}</span>')
            else:
                html_parts.append(clean_token)
        
        html_text = ''.join(html_parts)
        
        # Wrap in a styled div
        return f"""
        <div style="
            padding: 20px; 
            background-color: #1a1a1a; 
            border-radius: 10px; 
            border: 1px solid #444;
            line-height: 1.8;
            font-family: 'Courier New', monospace;
            word-wrap: break-word;
        ">
            <p style="color: #00d4ff; margin-bottom: 10px;">
                <strong>🎯 Token Importance Visualization</strong>
            </p>
            <p style="color: #888; font-size: 0.9em; margin-bottom: 15px;">
                Darker red = higher importance for generating the summary
            </p>
            {html_text}
        </div>
        """
    
    def generate_explainability_report(self, text: str, summary: str) -> Dict:
        """
        Generate comprehensive explainability report with all visualizations.
        """
        report = {}
        
        # Get attention weights
        with st.spinner("🔍 Analyzing attention patterns..."):
            attention_data = self.get_attention_weights(text, summary)
            if attention_data:
                report['attention_heatmap'] = self.create_attention_heatmap(attention_data)
            else:
                st.info("ℹ️ Attention visualization not available. Using alternative methods.")
        
        # Get sentence importance
        with st.spinner("📊 Calculating sentence importance..."):
            sentence_scores = self.calculate_sentence_importance(text, summary)
            if sentence_scores:
                report['sentence_chart'] = self.create_sentence_importance_chart(sentence_scores)
                report['sentence_scores'] = sentence_scores
        
        # Get token importance
        with st.spinner("🎯 Computing token importance..."):
            token_scores = self.get_token_importance_scores(text, summary)
            if token_scores:
                report['token_html'] = self.create_token_highlighting(text, token_scores)
                report['token_scores'] = token_scores
        
        return report
    
    def display_explainability_dashboard(self, text: str, summary: str):
        """
        Display complete explainability dashboard in Streamlit.
        """
        st.header("🔬 Explainability Dashboard")
        st.markdown("""
        <div style="padding: 10px; background-color: #1a1a1a; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: #00d4ff;">
                Understanding how BART created your summary through multiple lenses:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate report
        report = self.generate_explainability_report(text, summary)
        
        # Create tabs for different visualizations
        tabs = ["📊 Sentence Importance", "🎯 Token Importance", "📈 Summary Statistics"]
        
        # Only add Attention Heatmap tab if we have attention data
        if 'attention_heatmap' in report and report['attention_heatmap'] is not None:
            tabs.insert(2, "🔥 Attention Heatmap")
            tab1, tab2, tab3, tab4 = st.tabs(tabs)
        else:
            tab1, tab2, tab4 = st.tabs(tabs)
            tab3 = None  # No attention heatmap tab
        
        with tab1:
            if 'sentence_chart' in report:
                st.plotly_chart(report['sentence_chart'], use_container_width=True)
                
                with st.expander("📝 View all sentence scores"):
                    for i, (sent, score) in enumerate(report['sentence_scores'], 1):
                        if score > 0.1:
                            st.write(f"**{i}. Score: {score:.3f}**")
                            st.write(f"{sent[:200]}..." if len(sent) > 200 else sent)
                            st.divider()
        
        with tab2:
            if 'token_html' in report:
                st.markdown(report['token_html'], unsafe_allow_html=True)
                
                # Add legend
                st.markdown("""
                <div style="margin-top: 20px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;">
                    <p style="color: #888; font-size: 0.9em;">
                        <strong>Legend:</strong><br>
                        🔴 High importance (>0.7) | 
                        🟠 Medium importance (0.4-0.7) | 
                        🟡 Low importance (0.2-0.4)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        if tab3 is not None:
            with tab3:
                if 'attention_heatmap' in report and report['attention_heatmap'] is not None:
                    st.plotly_chart(report['attention_heatmap'], use_container_width=True)
                    st.info("""
                    **How to read this:**
                    - Y-axis: tokens in the generated summary
                    - X-axis: tokens from the input text  
                    - Brighter colors = higher attention
                    - Shows which input tokens the model "looked at" when generating each summary token
                    """)
        
        with (tab4 if tab3 is not None else tab4):
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
                    st.warning("📋 Highly Extractive Summary")
                elif extraction_ratio > 40:
                    st.info("🔄 Balanced Extractive-Abstractive")
                else:
                    st.success("✨ Highly Abstractive Summary")