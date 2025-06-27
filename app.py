import streamlit as st
import pandas as pd
import altair as alt
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, pipeline
import model 
import os
import math # Added for sqrt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import numpy as np # For confusion matrix display if needed
import re # For tokenization in word frequency
from collections import Counter # For word frequency
from PIL import Image

os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'
# Page config
st.set_page_config(
    page_title="Hybrid Recommender System For Product Recommendations",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display as a small logo + title on the same row
logo = Image.open("cute3d.png")
col_logo, col_title = st.columns([1, 10], gap="small")
with col_logo:
    st.image(logo, width=200)  # adjust width as needed
with col_title:
    st.markdown(
        '<div class="main-title" style="text-align: left; margin: 0;">'
        'Hybrid Recommender System For Product Recommendations'
        '</div>',
        unsafe_allow_html=True
    )
# Navigation buttons
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    if st.button("🏠 Home", key="home", use_container_width=True):
        st.session_state.page = "Main Page"

with col2:
    if st.button("📁 View Dataset", key="dataset", use_container_width=True):
        st.session_state.page = "View Dataset"

with col3:
    if st.button("🛍️ Recommendation System", key="recommender", use_container_width=True):
        st.session_state.page = "Recommender System"

with col4:
    if st.button("📈 Performance", key="performance", use_container_width=True):
        st.session_state.page = "📈 Model Performance"

# Initialize session state for page if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = "Main Page"

# Use session state for navigation
app_mode = st.session_state.page

# --- Load Data ---
@st.cache_resource
def load_all_data():
    """Load all required data and models."""
    try:
        # Get the absolute path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'cleaned_merged.csv')
        
        # Load the main dataset
        loaded_df = pd.read_csv(csv_path)
        
        # Create product summary map & Filter DataFrame
        item_text_map = (
            loaded_df
            .drop_duplicates(subset=['ProductId'])
            .set_index('ProductId')['CleanedSummary']
            .to_dict()
        )
        loaded_df = loaded_df[
            (~loaded_df['ProductId'].astype(str).str.isdigit()) &
            (~loaded_df['UserId'].astype(str).str.startswith('#oc'))
        ]
        
        # Get available product IDs and user IDs
        available_product_ids = sorted(list(loaded_df['ProductId'].unique()))
        available_user_ids = sorted(list(loaded_df['UserId'].unique()))
        
        # Load sentiment analyzer
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Load recommendation models
        model_loaded = model.load_models_and_data()
        
        return True, available_product_ids, available_user_ids, item_text_map, loaded_df, sentiment_analyzer, model_loaded
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return False, None, None, None, None, None, False


# --- Execute Loading with Spinner ---
with st.spinner('Loading data and models... Please wait...'):
    load_successful, available_product_ids, available_user_ids, product_summary_map, all_reviews_df, sentiment_analyzer, model_loaded = load_all_data()

if app_mode == "Main Page":
    st.markdown("""
    #### 🤖 System Overview
    This web application leverages **state-of-the-art machine learning** to provide smart, personalized product recommendations.  
    It combines:
    - <b>Collaborative Filtering (CF):</b> Learns from user behavior and preferences.
    - <b>Content-Based Filtering (CBF):</b> Understands product features and review sentiment.
    - <b>Hybrid Approach:</b> Merges both for more accurate and robust recommendations.
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("✨ Key Features")
    st.markdown("""
    - **Personalized Recommendations:** Get suggestions tailored to your preferences.
    - **Sentiment Analysis:** See how products are perceived by real users.
    - **Performance Metrics:** Transparent model evaluation and accuracy.
    - **Interactive Dashboard:** Easy navigation and clear visualizations.
    """)

elif app_mode == "View Dataset":
    st.title("🔍 Dataset")

    st.markdown("""
    **About the Dataset**

    The dataset shown below is a preprocessed from the original Amazon Product Review dataset.
    Key preprocessing steps include:
    - Removal of duplicates and irrelevant columns
    - Text cleaning and normalization (lowercasing, punctuation removal, etc.)
    - Tokenization and BERT-based embedding generation for summaries

    **Main columns include:**
    - `ProductId`, `UserId`: Unique identifiers for products and users
    - `Score`: User rating (1-5)
    - `Summary`, `CleanedSummary`: Original and cleaned review summaries
    - `Label`: Sentiment class derived from the score
    - `BERT_Text`: BERT embedding vector for each summary

    This cleaned dataset is used for both collaborative and content-based recommendation, as well as sentiment analysis.
    """)

    if load_successful:
        # Define dropped columns
        columns_designated_as_dropped = ['Unnamed: 0', 'Ori Text', 'Sentiment','Tokens', 'Input IDs']
        existing_dropped_columns = [col for col in columns_designated_as_dropped if col in all_reviews_df.columns]

        # Drop columns from main DataFrame
        df_cleaned = all_reviews_df.drop(columns=existing_dropped_columns, errors='ignore')

        # Dataset Info
        st.header("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df_cleaned.shape[0])
        with col2:
            st.metric("Columns", df_cleaned.shape[1])

        st.header("Dataset Preview")
        preview_rows = st.slider("Number of rows to display", 10, min(393560, len(df_cleaned)), 20)
        st.dataframe(df_cleaned.head(preview_rows), use_container_width=True)
    else:
        st.error("Failed to load dataset. Please check if the data file exists and is properly formatted.")

elif app_mode == "Recommender System":
    st.title("🛍️ Recommendation System")
    st.markdown("""
    ### Recommendation System
    This section provides three different ways to get product recommendations:
    1. **Product Recommendations**: Find similar products based on content
    2. **Sentiment Analysis**: Analyze the sentiment of product reviews
    3. **Hybrid Recommendations**: Get personalized recommendations combining user preferences and product similarity
    """)
    
    if load_successful and model_loaded:
        # Initialize selected_product_id in session state if it doesn't exist or data reloaded
        # This should be done BEFORE tabs are created and try to access/set it via selectbox
        if 'selected_product_id' not in st.session_state or not available_product_ids or st.session_state.selected_product_id not in available_product_ids:
            st.session_state.selected_product_id = available_product_ids[0] if available_product_ids else None

        # Create tabs for different recommendation types
        tab1, tab2, tab3 = st.tabs([
            "🛍️ Product Recommendations",
            "🔍 Sentiment Analysis",
            "🔄 Hybrid Recommendations"
        ])
        
        # Tab 1: Product Recommendations
        with tab1:
            # --- Product Selection moved into Tab 1 ---
            st.markdown("---")
            st.header("⚙️ Product Selection")
            if available_product_ids:
                # Use a callback to update session_state when selectbox changes
                def update_selected_product():
                    st.session_state.selected_product_id = st.session_state.product_select_key # product_select_key is the key of the selectbox
                
                st.selectbox(
                    "Select Product ID for Recommendations:",
                    options=available_product_ids,
                    key='product_select_key', 
                    index=available_product_ids.index(st.session_state.selected_product_id) if st.session_state.selected_product_id in available_product_ids else 0,
                    on_change=update_selected_product,
                    help="Choose a product to get recommendations."
                )
            else:
                st.error("No products available for selection.")
            st.markdown("---")
            
            st.markdown("""
            ### Content-Based Recommendations
            This tab helps you find products similar to the one you've selected. It uses Summary review content and ratings.            
            The recommendations are based on content similarity, so you'll see products that are most similar to the selected product.
            """)

            # Use selected_product_id from session_state
            current_selected_product_id_for_tab1 = st.session_state.get('selected_product_id')

            if current_selected_product_id_for_tab1:
                # Display product information
                st.subheader(f"Product Information: {current_selected_product_id_for_tab1}")
                product_summary = product_summary_map.get(current_selected_product_id_for_tab1, "No summary available")
                st.markdown(f"> {product_summary}")
                
                # Get content-based recommendations
                with st.spinner("Finding similar products..."):
                    try:
                        recommended_items, similarity_scores = model.get_cbf_recommendations(current_selected_product_id_for_tab1, k=5)
                        
                        if recommended_items:
                            st.subheader("Similar Products")
                            st.markdown("""
                            Below are the most similar products to your selection, ranked by similarity score.
                            """)
                            rec_df = pd.DataFrame({
                                'Product ID': recommended_items,
                                'Similarity Score': [f"{score:.4f}" for score in similarity_scores],
                                'Product Summary': [product_summary_map.get(item_id, "No summary available") for item_id in recommended_items]
                            })
                            st.dataframe(rec_df, use_container_width=True)
                        else:
                            st.info("No similar products found.")
                    except Exception as e:
                        st.error(f"Error getting recommendations: {e}")
            elif available_product_ids: # If no product selected but products are available
                st.info("Please select a Product ID above to see recommendations.")
        
        # Tab 2: Sentiment Analysis
        with tab2:
            st.header("Sentiment Analysis")
            st.markdown("""
            ### Review Sentiment Analysis
            """)
            
            # Use selected_product_id from session_state
            current_selected_product_id_for_tab2 = st.session_state.get('selected_product_id')

            if current_selected_product_id_for_tab2: # Ensure a product is selected
                # --- Display Custom CBF Model's Sentiment for the PRODUCT ---
                cbf_prerequisites_met = (
                    model.cbf_model is not None and
                    hasattr(model, 'item_embeddings') and 
                    model.item_embeddings is not None
                )

                if cbf_prerequisites_met:
                    st.subheader("Overall Product Sentiment (from Custom CBF Model)")
                    if current_selected_product_id_for_tab2 in model.item_embeddings:
                        try:
                            item_embedding = model.item_embeddings[current_selected_product_id_for_tab2].to(model.device)
                            if item_embedding.ndim == 1:
                                item_embedding = item_embedding.unsqueeze(0)

                            # Assuming model.cbf_model.fc1.in_features exists as used in performance tab
                            # It's safer to check if fc1 exists before accessing in_features
                            if hasattr(model.cbf_model, 'fc1') and hasattr(model.cbf_model.fc1, 'in_features'):
                                expected_dim = model.cbf_model.fc1.in_features
                                if item_embedding.shape[-1] == expected_dim:
                                    with torch.no_grad():
                                        model.cbf_model.eval()
                                        logits = model.cbf_model(item_embedding)
                                        # Apply Softmax to get probabilities
                                        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                                        predicted_label_idx = torch.argmax(logits, dim=1).item()
                                    
                                    cbf_sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
                                    product_cbf_sentiment = cbf_sentiment_map.get(predicted_label_idx, "Error predicting")
                                    
                                    # Display final prediction with color
                                    if product_cbf_sentiment == "POSITIVE":
                                        st.success(f"Predicted Product Sentiment: **{product_cbf_sentiment}**")
                                    elif product_cbf_sentiment == "NEGATIVE":
                                        st.error(f"Predicted Product Sentiment: **{product_cbf_sentiment}**")
                                    elif product_cbf_sentiment == "NEUTRAL":
                                        st.info(f"Predicted Product Sentiment: **{product_cbf_sentiment}**")
                                    else: # Error case
                                        st.warning(f"Predicted Product Sentiment: {product_cbf_sentiment}")

                                    # --- Compare with Average Review Score Sentiment ---
                                    st.markdown("**Comparison with Average Review Score:**")
                                    if all_reviews_df is not None and not all_reviews_df.empty:
                                        product_actual_reviews = all_reviews_df[all_reviews_df['ProductId'] == current_selected_product_id_for_tab2]
                                        if not product_actual_reviews.empty and 'Score' in product_actual_reviews.columns:
                                            # Ensure 'Score' is numeric, coercing errors
                                            product_actual_reviews['Score'] = pd.to_numeric(product_actual_reviews['Score'], errors='coerce')
                                            average_score = product_actual_reviews['Score'].mean()
                                            
                                            if pd.notna(average_score):
                                                # Define sentiment based on average score (same logic as training if possible)
                                                avg_score_sentiment = ""
                                                if average_score < 3:
                                                    avg_score_sentiment = "NEGATIVE"
                                                elif average_score == 3:
                                                    avg_score_sentiment = "NEUTRAL"
                                                else: # average_score > 3
                                                    avg_score_sentiment = "POSITIVE"
                                                
                                                st.markdown(f"- Average Review Score for this Product: **{average_score:.2f}**")
                                                st.markdown(f"- Sentiment based on Average Score: **{avg_score_sentiment}**")
                                                
                                            else:
                                                st.info("Could not calculate a valid average review score for this product (e.g., no numeric scores found).")
                                        else:
                                            st.info("No reviews or 'Score' column found for this product to calculate average score sentiment.")
                                    else:
                                        st.info("Review data is not available for average score comparison.")

                                else:
                                    st.warning(f"Custom CBF Model: Embedding dimension mismatch for product {current_selected_product_id_for_tab2}. Expected {expected_dim}, got {item_embedding.shape[-1]}. Cannot predict overall sentiment.")
                            else:
                                st.warning("Custom CBF Model: Could not determine expected input dimension (e.g., `fc1.in_features` not found or `fc1` layer missing). Cannot predict overall sentiment.")
                        except Exception as e_cbf_prod_sentiment:
                            st.error(f"Error predicting product sentiment with Custom CBF Model: {e_cbf_prod_sentiment}")
                    else:
                        st.info("Custom CBF Model: This product does not have a pre-computed embedding for overall sentiment prediction.")
                    st.markdown("---") # Separator
                else:
                    # This warning appears if model_loaded was True (so model.py's loading function was called),
                    # but cbf_model or item_embeddings are still None/missing.
                    st.warning(
                        "The Custom CBF model (`cbf_model.pth`) or its item embeddings (`item_embeddings.pt`) "
                        "appear to be missing or failed to load correctly via `model.py`. "
                        "Therefore, the overall product sentiment from your custom model cannot be displayed. "
                        "Please check the `load_models_and_data` function in `model.py` and ensure these files are "
                        "correctly loaded and assigned to `model.cbf_model` and `model.item_embeddings` respectively."
                    )

            if current_selected_product_id_for_tab2 and sentiment_analyzer:
                # Get product reviews
                product_reviews = all_reviews_df[all_reviews_df['ProductId'] == current_selected_product_id_for_tab2]
                
                if not product_reviews.empty:
                    # Find the text column (could be 'Text' or 'reviewText')
                    text_column = None
                    for col in ['Text', 'reviewText', 'CleanedSummary']:
                        if col in product_reviews.columns:
                            text_column = col
                            break
                                
                    if text_column:
                        # Analyze sentiment for first 20000 reviews
                        max_reviews = min(200000, len(product_reviews))
                        analysis_df = product_reviews.head(max_reviews).copy()
                        
                        with st.spinner("Analyzing review sentiments..."):
                            sentiments = []
                            scores = []
                            progress_bar = st.progress(0)
                                
                            for i, review in enumerate(analysis_df[text_column].fillna('').iloc[:max_reviews]):
                                if not review or review.isspace():
                                    sentiments.append("Neutral")
                                    scores.append(0.5)
                                    continue
                                        
                                try:
                                    result = sentiment_analyzer(review[:1024])
                                    sentiments.append(result[0]['label'])
                                    scores.append(result[0]['score'])
                                except:
                                    sentiments.append("Error")
                                    scores.append(0)
                                        
                                progress_bar.progress((i + 1) / max_reviews)
                                
                            progress_bar.empty()
                            
                            # Add results to dataframe
                            analysis_df['Sentiment'] = sentiments
                            analysis_df['Confidence'] = scores
                            
                        # Normalize sentiment labels to uppercase
                        analysis_df['Sentiment'] = analysis_df['Sentiment'].str.upper()
                        
                        # Display sentiment distribution
                        st.subheader("Sentiment Distribution")
                        st.markdown("""
                        This chart shows the distribution of sentiments across all analyzed reviews.
                        """)
                        sentiment_counts = analysis_df['Sentiment'].value_counts()
                        
                        chart = alt.Chart(sentiment_counts.reset_index()).mark_bar().encode(
                            x=alt.X('Sentiment:N', title='Sentiment'),
                            y=alt.Y('count:Q', title='Number of Reviews'),
                            color=alt.Color('Sentiment:N', scale=alt.Scale(
                                domain=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                                range=['#6bb373', '#bf4545', '#ebc75e'] 
                            )),
                            tooltip=['Sentiment', 'count']
                        ).properties(width=600, height=400)
                        
                        st.altair_chart(chart, use_container_width=True)

                        # --- Enhanced Sentiment Statistics ---
                        st.subheader("Sentiment Analysis Summary")
                        total_analyzed = len(analysis_df)
                        positive_count = sentiment_counts.get('POSITIVE', 0)
                        negative_count = sentiment_counts.get('NEGATIVE', 0)
                        neutral_count = sentiment_counts.get('NEUTRAL', 0)

                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Total Reviews Analyzed", total_analyzed)
                            positive_percentage = (positive_count / total_analyzed) if total_analyzed > 0 else 0.0
                            st.metric("Positive Reviews", f"{positive_count} ({positive_percentage:.1%})")
                        with col_stats2:
                            negative_percentage = (negative_count / total_analyzed) if total_analyzed > 0 else 0.0
                            st.metric("Negative Reviews", f"{negative_count} ({negative_percentage:.1%})")
                            neutral_percentage = (neutral_count / total_analyzed) if total_analyzed > 0 else 0.0
                            st.metric("Neutral Reviews", f"{neutral_count} ({neutral_percentage:.1%})")
                        with col_stats3:
                            avg_confidence_overall = analysis_df['Confidence'].mean() if not analysis_df.empty else 0
                            st.metric("Avg. Overall Confidence", f"{avg_confidence_overall:.2%}")

                        # Display reviews with sentiment
                        st.subheader("Reviews with Sentiment Analysis")
                        st.markdown("""
                        Below are the individual reviews with their sentiment analysis results. The confidence score indicates how certain the model is about the sentiment classification.
                        """)
                        display_columns = [text_column, 'Score', 'Sentiment', 'Confidence']
                        display_df = analysis_df[display_columns].copy()
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.error("No text column found for sentiment analysis")
                else:
                    st.info("No reviews found for this product.")
            else:
                st.info("Select a product in the 'Product Recommendations' tab to see its sentiment analysis.")
        
        # Tab 3: Hybrid Recommendations
        with tab3:
            st.header("Hybrid Recommendations")
            st.markdown("""
            ### Hybrid Recommendation System
            This tab combines two recommendation approaches:
            1. **Collaborative Filtering**: Uses patterns in user behavior to recommend products
            2. **Content-Based Filtering**: Uses product features and characteristics to find similar items
            """)
            
            # User ID selection
            selected_user_id = st.selectbox(
                "Select User ID",
                options=available_user_ids,
                index=0 if available_user_ids else -1,
                help="Choose a user to get personalized recommendations"
            )
            
            # Fixed weights for CF and CBF
            cf_weight = 0.5
            cbf_weight = 0.5
            
            # Number of recommendations
            num_recs = st.number_input(
                "Number of Recommendations",
                min_value=1, max_value=10000, value=10
            )
            
            # Use selected_product_id from session_state for the cbf part of hybrid
            current_selected_product_id_for_tab3 = st.session_state.get('selected_product_id')

            if st.button("Generate Hybrid Recommendations"):
                if not selected_user_id:
                    st.warning("Please select a User ID")
                elif not current_selected_product_id_for_tab3: 
                    st.warning("Please select a Product ID in the 'Product Recommendations' tab first.")
                else:
                    with st.spinner("Generating hybrid recommendations..."):
                        try:
                            top_n = 10000 
                            # Get hybrid recommendations - ensure model.get_hybrid_recommendations uses current_selected_product_id_for_tab3 if needed
                            items, hybrid_scores, cf_scores, cbf_scores = model.get_hybrid_recommendations(
                                selected_user_id,
                                available_product_ids,
                                model.item_embeddings,
                                n_recommendations=num_recs,
                                diversity_weight=0.5  # Using default value
                            )
                            
                            if items:
                                sampled_indices = np.random.choice(len(items), size=min(num_recs, len(items)), replace=False)
                                sampled_items = [items[i] for i in sampled_indices]
                                sampled_hybrid_scores = [hybrid_scores[i] for i in sampled_indices]
                                sampled_cf_scores = [cf_scores[i] for i in sampled_indices]
                                sampled_cbf_scores = [cbf_scores[i] for i in sampled_indices]

                                # Create display dataframe
                                hybrid_df = pd.DataFrame({
                                    'Product ID': sampled_items,
                                    'Hybrid Score': [f"{score:.4f}" for score in sampled_hybrid_scores],
                                    'CF Component': [f"{(score - 1)/4:.4f}" for score in sampled_cf_scores],
                                    'CBF Component': [f"{score:.4f}" for score in sampled_cbf_scores],
                                    'Product Summary': [product_summary_map.get(item_id, "No summary available") for item_id in sampled_items]
                                })
                                st.subheader("Hybrid Recommendations")
                                st.markdown("""
                                Below are the recommended products, combining both collaborative and content-based filtering approaches. 
                                The scores are normalized between 0 and 1, showing how each approach contributed to the final recommendation.
                                """)
                                st.dataframe(hybrid_df, use_container_width=True)
                                
                                # Visualize model contributions
                                st.subheader("Model Contribution Analysis")
                                st.markdown("""
                                This chart shows how each recommendation approach (Collaborative Filtering and Content-Based Filtering) 
                                contributed to the final recommendations. The scores are normalized between 0 and 1.
                                """)
                                
                                # Create visualization data
                                # 1) Build a wide DataFrame
                                df_contrib_wide = pd.DataFrame({
                                    'Product ID': items,
                                    'CF': [round((s-1.0)/4.0, 4) for s in cf_scores],
                                    'CBF': [round(s, 4)           for s in cbf_scores]
                                })

                                # 2) Melt into long form
                                contribution_df = df_contrib_wide.melt(
                                    id_vars='Product ID',
                                    var_name='Model',
                                    value_name='Contribution'
                                )

                                # 3) Plot grouped bars with tooltip
                                chart = (
                                    alt.Chart(contribution_df)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X('Product ID:N', title='Product ID'),
                                        y=alt.Y('Contribution:Q', title='Normalized Score', stack=None),
                                        color=alt.Color('Model:N', title='Model',
                                                        scale=alt.Scale(domain=['CF','CBF'],
                                                                        range=['#bf84b1','#e6c481'])),
                                        xOffset=alt.XOffset('Model:N'),
                                        tooltip=[
                                            alt.Tooltip('Product ID:N', title='Product'),
                                            alt.Tooltip('Model:N',      title='Model'),
                                            alt.Tooltip('Contribution:Q', title='Contribution', format='.4f'),
                                        ]
                                    )
                                    .properties(width=alt.Step(60))
                                )

                                st.altair_chart(chart, use_container_width=True)

                            else:
                                st.info("No hybrid recommendations could be generated")
                        except Exception as e:
                            st.error(f"""
                            Error generating recommendations: {str(e)}
                            
                            This could be due to:
                            1. Missing model files
                            2. Corrupted model files
                            3. Incompatible model versions
                            
                            Please ensure the following files are present:
                            - cf_model.pth
                            - cf_mappings.pkl
                            - item_embeddings.pt
                            """)
    else:
        if not load_successful:
            st.error("Data loading failed. Recommender system cannot be displayed.")
        elif not model_loaded:
            st.error("Model loading failed. Recommender system cannot be displayed.")
        else:
            st.info("Please select a Product ID from the sidebar to get started.")
            
elif app_mode == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")
    # Load your test data
    test_df = pd.read_csv('cf_test_data.csv')

    all_preds = []
    all_targets = []

    for idx, row in test_df.iterrows():
        user_id = row['UserId']
        item_id = row['ProductId']
        true_rating = row['Score']
        
        # Get hybrid score for this user-item pair
        recommended_items, hybrid_scores, cf_scores, cbf_scores = model.get_hybrid_recommendations(
            user_id=user_id,
            item_ids=[item_id],
            all_item_embeddings=model.item_embeddings,
            n_recommendations=1
        )
        if hybrid_scores:
            all_preds.append(hybrid_scores[0])
            all_targets.append(true_rating)

    # after collecting all_preds (0–1) and all_targets (1–5):
    all_preds_rescaled = [p * 4 + 1 for p in all_preds]   # maps 0→1, 1→5

    rmse_rescaled = np.sqrt(mean_squared_error(all_targets, all_preds_rescaled))
    mae_rescaled  = mean_absolute_error(all_targets, all_preds_rescaled)
    # Display metrics
    st.subheader("Hybrid Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse_rescaled:.4f}")
    col2.metric("MAE",  f"{mae_rescaled:.4f}")


    # Show a few sample predictions
    results_df = pd.DataFrame({
        "True Rating (1-5)":     all_targets,
        "Predicted Rating (1-5)": all_preds_rescaled,
        "Predicted Score (0-1)": all_preds,
    })
    st.markdown("**Sample Prediction vs. True Ratings**")
    st.dataframe(results_df.head(20), use_container_width=True)


    # --- Overall Model Comparison (70:30 vs 80:20 splits) ---
    st.subheader("🔍 Overall CBF Model Comparison")

    # 1) build the DataFrame
    data = {
        "Model": [
            "Logistic Regression",
            "Gradient Boost",
            "K-Nearest Neighbors",
            "XGBoost",
            "BERT"
        ],
        # 70:30
        "Accuracy_70": [0.6985, 0.6872, 0.8463, 0.7115, 0.8626],
        "Precision_70":[0.6967, 0.6871, 0.8625, 0.7113, 0.8600],
        "Recall_70":   [0.6985, 0.6872, 0.8463, 0.7115, 0.8600],
        "F1_70":       [0.6971, 0.6870, 0.8418, 0.7113, 0.8600],
        # 80:20
        "Accuracy_80": [0.6989, 0.6862, 0.8463, 0.7127, 0.8953],
        "Precision_80":[0.6972, 0.6862, 0.8625, 0.7126, 0.9000],
        "Recall_80":   [0.6989, 0.6862, 0.8463, 0.7127, 0.9000],
        "F1_80":       [0.6976, 0.6861, 0.8418, 0.7125, 0.9000],
    }
    metrics_df = pd.DataFrame(data)

    st.dataframe(
        metrics_df.style.format({c: "{:.4f}" for c in metrics_df.columns if c != "Model"}),
        use_container_width=True
    )

    # Melt your 70:30 metrics into long form:
    df_70 = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy_70","Precision_70","Recall_70","F1_70"],
        var_name="Metric",
        value_name="Score"
    )
    df_70["Metric"] = df_70["Metric"].str.replace("_70", "")

    chart_70_grouped = (
        alt.Chart(df_70)
        .mark_bar()
        .encode(
            x=alt.X('Metric:N', title='Metric'),
            y=alt.Y('Score:Q', title='Score', stack=None),
            color=alt.Color('Model:N', title='Model'),
            xOffset=alt.XOffset('Model:N'),
            tooltip=[
                alt.Tooltip('Model:N', title='Model'),
                alt.Tooltip('Metric:N', title='Metric'),
                alt.Tooltip('Score:Q', title='Score', format='.4f')
            ]
        )
        .properties(
            title='Performance Comparison (70:30 Train-Test Split)',
            width=600,
            height=400
        )
    )
    st.altair_chart(chart_70_grouped, use_container_width=True)

    # Melt your 80:20 metrics into long form:
    df_80 = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy_80","Precision_80","Recall_80","F1_80"],
        var_name="Metric",
        value_name="Score"
    )
    df_80["Metric"] = df_80["Metric"].str.replace("_80", "")

    chart_80_grouped = (
        alt.Chart(df_80)
        .mark_bar()
        .encode(
            x=alt.X('Metric:N', title='Metric'),
            y=alt.Y('Score:Q', title='Score', stack=None),
            color=alt.Color('Model:N', title='Model'),
            xOffset=alt.XOffset('Model:N'),
            tooltip=[
                alt.Tooltip('Model:N', title='Model'),
                alt.Tooltip('Metric:N', title='Metric'),
                alt.Tooltip('Score:Q', title='Score', format='.4f')
            ]
        )
        .properties(
            title='Performance Comparison (80:20 Train-Test Split)',
            width=600,
            height=400
        )
    )
    st.altair_chart(chart_80_grouped, use_container_width=True)

# --- Footer ---
st.markdown("---") 
st.markdown(
    """
    <p style="text-align: center; 
              color: #B1B1B1; 
              font-size: 0.9rem;
              margin-top: 1rem;">
      Tan Xin Yi | 1211100903 | Project ID: FYP02-DS-T2510-0044
    </p>
    """,
    unsafe_allow_html=True
)