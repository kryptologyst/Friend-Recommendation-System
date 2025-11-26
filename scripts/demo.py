#!/usr/bin/env python3
"""Streamlit demo for friend recommendation system."""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader
from models import BaseRecommender
from utils import load_config
from utils.metrics import RecommendationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Friend Recommendation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data with caching."""
    try:
        data_dir = Path(__file__).parent.parent / "data"
        loader = DataLoader(str(data_dir))
        
        users_df = loader.load_users()
        interactions_df = loader.load_interactions()
        items_df = loader.load_items()
        
        return users_df, interactions_df, items_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_data
def load_models():
    """Load trained models with caching."""
    try:
        models_dir = Path(__file__).parent.parent / "models"
        models = {}
        
        for model_file in models_dir.glob("*.pkl"):
            if model_file.name == "training_metadata.pkl":
                continue
            
            model_name = model_file.stem
            model = BaseRecommender.load_model(str(model_file))
            models[model_name] = model
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


@st.cache_data
def load_evaluation_results():
    """Load evaluation results with caching."""
    try:
        results_dir = Path(__file__).parent.parent / "results"
        results_file = results_dir / "evaluation_results.csv"
        
        if results_file.exists():
            return pd.read_csv(results_file, index_col=0)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        return None


def display_user_info(user_id: int, users_df: pd.DataFrame):
    """Display user information."""
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    
    st.markdown("### User Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("User ID", user_id)
        st.metric("Name", user_info['name'])
    
    with col2:
        st.metric("Age", user_info['age'])
        st.metric("Location", user_info['location'])
    
    with col3:
        st.metric("Occupation", user_info['occupation'])
    
    st.markdown("**Interests:**")
    st.write(user_info['interests'])


def display_recommendations(user_id: int, model_name: str, models: Dict[str, BaseRecommender], 
                          users_df: pd.DataFrame, n_recommendations: int = 10):
    """Display recommendations for a user."""
    if model_name not in models:
        st.error(f"Model {model_name} not found!")
        return
    
    model = models[model_name]
    
    try:
        recommendations = model.predict(user_id, n_recommendations)
        
        st.markdown(f"### Recommendations from {model_name}")
        
        if not recommendations:
            st.warning("No recommendations available for this user.")
            return
        
        for i, friend_id in enumerate(recommendations, 1):
            friend_info = users_df[users_df['user_id'] == friend_id].iloc[0]
            
            with st.container():
                st.markdown(f"**{i}. {friend_info['name']}** (ID: {friend_id})")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Age: {friend_info['age']}")
                with col2:
                    st.write(f"Location: {friend_info['location']}")
                with col3:
                    st.write(f"Occupation: {friend_info['occupation']}")
                
                st.write(f"**Interests:** {friend_info['interests']}")
                
                # Calculate similarity score if available
                try:
                    similarity = model.get_similarity_score(user_id, friend_id)
                    st.write(f"**Similarity Score:** {similarity:.3f}")
                except:
                    pass
                
                st.divider()
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")


def display_model_comparison(user_id: int, models: Dict[str, BaseRecommender], 
                           users_df: pd.DataFrame, n_recommendations: int = 5):
    """Display recommendations from multiple models for comparison."""
    st.markdown("### Model Comparison")
    
    model_names = list(models.keys())
    selected_models = st.multiselect(
        "Select models to compare:",
        model_names,
        default=model_names[:3] if len(model_names) >= 3 else model_names
    )
    
    if not selected_models:
        st.warning("Please select at least one model to compare.")
        return
    
    # Create tabs for each model
    tabs = st.tabs(selected_models)
    
    for i, model_name in enumerate(selected_models):
        with tabs[i]:
            display_recommendations(user_id, model_name, models, users_df, n_recommendations)


def display_evaluation_dashboard(results_df: pd.DataFrame):
    """Display evaluation metrics dashboard."""
    st.markdown("### Model Performance Dashboard")
    
    if results_df is None:
        st.warning("No evaluation results available. Please run the evaluation script first.")
        return
    
    # Select metrics to display
    available_metrics = [col for col in results_df.columns if '@' in col]
    selected_metrics = st.multiselect(
        "Select metrics to display:",
        available_metrics,
        default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=len(selected_metrics), 
        cols=1,
        subplot_titles=selected_metrics,
        vertical_spacing=0.1
    )
    
    for i, metric in enumerate(selected_metrics, 1):
        fig.add_trace(
            go.Bar(
                x=results_df.index,
                y=results_df[metric],
                name=metric,
                showlegend=False
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(selected_metrics),
        title_text="Model Performance Comparison",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed metrics table
    st.markdown("### Detailed Metrics")
    st.dataframe(results_df.round(4))


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">👥 Friend Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    users_df, interactions_df, items_df = load_data()
    
    if users_df is None:
        st.error("Failed to load data. Please ensure data files exist.")
        return
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No trained models found. Please train models first.")
        return
    
    # Load evaluation results
    results_df = load_evaluation_results()
    
    # Sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["Recommendations", "Model Comparison", "Evaluation Dashboard", "Data Overview"]
    )
    
    # Main content based on selected page
    if page == "Recommendations":
        st.markdown("## Get Friend Recommendations")
        
        # User selection
        user_ids = users_df['user_id'].tolist()
        selected_user = st.selectbox("Select a user:", user_ids)
        
        # Model selection
        model_names = list(models.keys())
        selected_model = st.selectbox("Select a model:", model_names)
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 1, 20, 10)
        
        # Display user info
        display_user_info(selected_user, users_df)
        
        # Display recommendations
        display_recommendations(selected_user, selected_model, models, users_df, n_recs)
    
    elif page == "Model Comparison":
        st.markdown("## Compare Models")
        
        # User selection
        user_ids = users_df['user_id'].tolist()
        selected_user = st.selectbox("Select a user:", user_ids, key="comparison_user")
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 1, 10, 5, key="comparison_recs")
        
        # Display user info
        display_user_info(selected_user, users_df)
        
        # Display model comparison
        display_model_comparison(selected_user, models, users_df, n_recs)
    
    elif page == "Evaluation Dashboard":
        st.markdown("## Model Performance Dashboard")
        display_evaluation_dashboard(results_df)
    
    elif page == "Data Overview":
        st.markdown("## Data Overview")
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        with col2:
            st.metric("Total Interactions", len(interactions_df))
        with col3:
            st.metric("Total Items", len(items_df))
        with col4:
            sparsity = 1 - (len(interactions_df) / (len(users_df) * len(users_df)))
            st.metric("Sparsity", f"{sparsity:.3f}")
        
        # User distribution by age
        st.markdown("### User Age Distribution")
        fig_age = px.histogram(users_df, x='age', nbins=20, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
        
        # User distribution by location
        st.markdown("### User Location Distribution")
        location_counts = users_df['location'].value_counts()
        fig_location = px.pie(values=location_counts.values, names=location_counts.index, title="Location Distribution")
        st.plotly_chart(fig_location, use_container_width=True)
        
        # Interaction patterns
        st.markdown("### Interaction Patterns")
        
        # Interactions per user
        user_interactions = interactions_df.groupby('user_id').size()
        fig_interactions = px.histogram(x=user_interactions.values, nbins=20, title="Interactions per User")
        st.plotly_chart(fig_interactions, use_container_width=True)
        
        # Sample data tables
        st.markdown("### Sample Data")
        
        tab1, tab2, tab3 = st.tabs(["Users", "Interactions", "Items"])
        
        with tab1:
            st.dataframe(users_df.head(10))
        
        with tab2:
            st.dataframe(interactions_df.head(10))
        
        with tab3:
            st.dataframe(items_df.head(10))


if __name__ == "__main__":
    main()
