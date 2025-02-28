# Mental Health Self-Analysis Tool - README

## Overview

This Streamlit application provides a Mental Health Self-Analysis Tool that allows users to input text describing their feelings, thoughts, or symptoms. The tool then analyzes the input using a combination of machine learning models (Random Forest, LDA, and BERT) to identify potential mental health patterns and provide relevant recommendations.

**Key Features:**

-   **Text Analysis:** Processes user input to identify patterns related to common mental health concerns.
-   **Topic Identification:** Uses a trained Random Forest model to predict the most relevant mental health topic.
-   **BERT Embeddings:** Leverages BERT for advanced text feature extraction.
-   **LDA Topic Modeling:** Utilizes LDA to provide topic descriptions.
-   **Personalized Recommendations:** Offers tailored recommendations based on the identified topic.
-   **User-Friendly Interface:** Built with Streamlit for easy deployment and use.
-   **Offline Model Loading:** Supports loading models from local cache for faster performance.
-   **Progress Indicators:** Provides progress bars and spinners to enhance user experience during model loading and analysis.
-   **Clear Disclaimers and Resources:** Includes important disclaimers and links to mental health resources.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install streamlit transformers torch numpy scikit-learn joblib
    ```

4.  **Download the BERT model (if not already cached):**

    The application attempts to load the `bert-base-uncased` model from a local cache (`./model_cache`). If the model is not found, it will be downloaded automatically. You can also manually download it:

    ```bash
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./model_cache')
    model = AutoModel.from_pretrained('bert-base-uncased', cache_dir='./model_cache')
    ```

5.  **Place your pre-trained models:**

    Ensure the following files are in your project directory:

    -   `rf_model.joblib` (Random Forest model)
    -   `vectorizer.joblib` (Vectorizer model)
    -   `lda_model.joblib` (LDA model)

    If you don't have these models, you will need to train them using a suitable dataset.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    (Replace `app.py` with the name of your Streamlit script.)

2.  **Open the application in your web browser:**

    Streamlit will provide a local URL (e.g., `http://localhost:8501`). Open this URL in your browser.

3.  **Enter your text:**

    In the text area, describe your feelings, thoughts, or symptoms.

4.  **Click "Analyze":**

    The application will process your input and display the analysis results, including the identified topic and recommendations.

## Model Training (If you need to train your own models)

To train the Random Forest, Vectorizer, and LDA models, you will need a suitable dataset of text data labeled with mental health topics. Here's a general outline:

1.  **Data Preparation:**

    -   Collect and preprocess your dataset.
    -   Split the data into training and testing sets.

2.  **Text Vectorization:**

    -   Use `TfidfVectorizer` or a similar technique to convert text data into numerical vectors.
    -   Save the trained vectorizer using `joblib.dump(vectorizer, 'vectorizer.joblib')`.

3.  **LDA Model Training:**

    -   Train an LDA model on the vectorized data.
    -   Save the trained LDA model using `joblib.dump(lda_model, 'lda_model.joblib')`.

4.  **Feature Extraction:**

    -   Use the trained BERT model to extract features from the text data.
    -   Create a dataset with BERT features and corresponding labels.

5.  **Random Forest Model Training:**

    -   Train a Random Forest classifier on the BERT features.
    -   Save the trained Random Forest model using `joblib.dump(rf_model, 'rf_model.joblib')`.

## Customization

-   **Topic Descriptions:** Modify the `get_topic_description()` function to customize the descriptions of mental health topics.
-   **Recommendations:** Update the `get_recommendations()` function to provide different recommendations based on the identified topics.
-   **Styling:** Customize the appearance of the Streamlit app by modifying the CSS styles in the `st.markdown()` blocks.
-   **Model Selection:** Experiment with different machine learning models or BERT variants to improve performance.


This tool is for educational purposes only and should not replace professional medical advice. If you are experiencing severe mental health symptoms, please consult with a qualified mental health professional.
## YouTube Video

For a visual walkthrough and demonstration of this application, please watch the following YouTube video:


## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

