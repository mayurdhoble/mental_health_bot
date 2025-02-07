import streamlit as st
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re
import os
import joblib
# Load the saved models
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with st.spinner("Loading models... This may take a few minutes on first run."):
            progress_text = "Loading models..."
            progress_bar = st.progress(0)
            
            # Check if model files exist
            if not os.path.exists('rf_model.joblib'):
                st.error("Random Forest model file not found!")
                return None, None, None, None, None
            progress_bar.progress(20)
            
            if not os.path.exists('vectorizer.joblib'):
                st.error("Vectorizer file not found!")
                return None, None, None, None, None
            progress_bar.progress(30)
            
            if not os.path.exists('lda_model.joblib'):
                st.error("LDA model file not found!")
                return None, None, None, None, None
            progress_bar.progress(40)

            # Load the models using joblib
            rf_model = joblib.load('rf_model.joblib')
            progress_bar.progress(50)
            vectorizer = joblib.load('vectorizer.joblib')
            progress_bar.progress(60)
            lda_model = joblib.load('lda_model.joblib')
            progress_bar.progress(70)

            # Load BERT models with offline mode if available
            try:
                st.info("Loading models... This may take a few minutes on first run.")
                # Try to load from cache first
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', 
                                                        local_files_only=True,
                                                        cache_dir='./model_cache')
                progress_bar.progress(80)
                
                bert_model = AutoModel.from_pretrained('bert-base-uncased', 
                                                     local_files_only=True,
                                                     cache_dir='./model_cache')
                progress_bar.progress(90)
                
            except Exception as e:
                # If not in cache, download them
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',
                                                        cache_dir='./model_cache')
                bert_model = AutoModel.from_pretrained('bert-base-uncased',
                                                     cache_dir='./model_cache')
            
            progress_bar.progress(100)
            progress_bar.empty()
            
            return rf_model, vectorizer, lda_model, tokenizer, bert_model

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

        

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def extract_bert_features(text, tokenizer, bert_model):
    if tokenizer is None or bert_model is None:
        raise ValueError("BERT models not properly loaded")
        
    # Tokenize and encode text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use [CLS] token embedding as feature vector
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embedding[0]

def get_topic_description(topic_num):
    # Define descriptions for each topic
    topic_descriptions = {
        0: "Anxiety and Stress Related Symptoms",
        1: "Depression and Mood Related Issues",
        2: "Sleep and Energy Problems",
        3: "Self-esteem and Personal Worth Issues",
        4: "Social and Relationship Concerns"
    }
    return topic_descriptions.get(topic_num, "Unknown Topic")

def get_recommendations(topic_num):
    recommendations = {
        0: [
            "Practice deep breathing exercises",
            "Try meditation or mindfulness",
            "Consider talking to a therapist",
            "Maintain a regular exercise routine",
            "Limit caffeine and alcohol intake"
        ],
        1: [
            "Establish a daily routine",
            "Set small, achievable goals",
            "Reach out to friends or family",
            "Consider professional help",
            "Engage in activities you once enjoyed"
        ],
        2: [
            "Maintain a consistent sleep schedule",
            "Create a relaxing bedtime routine",
            "Limit screen time before bed",
            "Exercise regularly (but not close to bedtime)",
            "Consider sleep hygiene practices"
        ],
        3: [
            "Practice positive self-talk",
            "List your achievements and strengths",
            "Set boundaries in relationships",
            "Engage in self-care activities",
            "Join support groups"
        ],
        4: [
            "Practice active listening",
            "Join social groups or clubs",
            "Improve communication skills",
            "Set healthy boundaries",
            "Seek relationship counseling if needed"
        ]
    }
    return recommendations.get(topic_num, ["Please consult with a mental health professional for personalized advice."])

def main():
    st.set_page_config(
        page_title="Mental Health Self-Analysis Tool",
        page_icon="🧠",
        layout="wide"
    )

    # Load models
    rf_model, vectorizer, lda_model, tokenizer, bert_model = load_models()

    # Check if models loaded successfully
    if None in (rf_model, vectorizer, lda_model, tokenizer, bert_model):
        st.error("Failed to load one or more models. Please check the model files and try again.")
        return

    st.title("🧠 Mental Health Self-Analysis Tool")
    
    # Use custom CSS to ensure text visibility
    st.markdown("""
        <style>
        .stMarkdown {
            color: #000000 !important;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

    # Welcome message with custom styling
    st.markdown("""
        <div class="info-box">
            <h4 style="color: #000000;">Welcome to the Mental Health Self-Analysis Tool</h4>
            <p style="color: #000000;">Please describe your feelings, thoughts, or symptoms below. This tool will analyze your input and provide potential insights.</p>
            <p style="color: #000000;"><strong>Note:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

    # User input with explicit styling
    st.markdown('<p style="color: #000000;">Describe your symptoms or feelings:</p>', unsafe_allow_html=True)
    # User input
    user_input = st.text_area(
        "Describe your symptoms or feelings:",
        height=150,
        placeholder="Example: I've been feeling overwhelmed lately, having trouble sleeping, and finding it hard to concentrate..."
    )
    if st.button("Analyze", type="primary"):
        if user_input:
            try:
                with st.spinner("Analyzing your input..."):
                    # Preprocess input
                    processed_input = preprocess_text(user_input)
                    
                    # Extract BERT features
                    features = extract_bert_features(processed_input, tokenizer, bert_model)
                    
                    # Make prediction
                    prediction = rf_model.predict([features])[0]
                    
                    # Get topic description and recommendations
                    topic_desc = get_topic_description(prediction)
                    recommendations = get_recommendations(prediction)

                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background-color: #e1f5fe; padding: 20px; border-radius: 5px;'>
                        <h3 style="color: #000000;">Analysis Results</h3>
                        <p style="color: #000000;"><strong>Primary Pattern Identified:</strong> {topic_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown("""
                        <div style='background-color: #f3e5f5; padding: 20px; border-radius: 5px;'>
                        <h3 style="color: #000000;">Suggested Steps</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        for rec in recommendations:
                            st.markdown(f'<p style="color: #000000;">• {rec}</p>', unsafe_allow_html=True)

                    # Update disclaimer styling
                    st.markdown("""
                    <div style='background-color: #ffebee; padding: 10px; border-radius: 5px; margin-top: 20px;'>
                    <p style="color: #000000;"><strong>Important Disclaimer:</strong> This analysis is not a medical diagnosis. If you're experiencing severe symptoms or having thoughts of self-harm, please seek immediate professional help.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Update resources styling
                    st.markdown("""
                    <div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px;'>
                    <h4 style="color: #000000;">Helpful Resources:</h4>
                    <ul style="color: #000000;">
                        <li>National Crisis Hotline: 988</li>
                        <li>Crisis Text Line: Text HOME to 741741</li>
                        <li>SAMHSA's National Helpline: 1-800-662-4357</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please make sure all models are properly loaded and try again.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()