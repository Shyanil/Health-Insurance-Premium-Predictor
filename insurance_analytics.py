import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    """Load and enhance the insurance dataset."""
    df = pd.read_csv('insurance.csv')
    
    # Add derived features for better analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                            labels=['Young Adult', 'Adult', 'Middle Age', 'Senior'])
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    return df

def main():
    st.set_page_config(layout="wide")
    
    # Custom styling
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Advanced Insurance Data Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar for analysis controls
    st.sidebar.header("Analysis Controls")
    
    # Variable selection for analysis
    primary_var = st.sidebar.selectbox(
        "Select Primary Variable",
        ['age', 'bmi', 'children', 'charges', 'region', 'smoker', 'sex']
    )
    
    secondary_var = st.sidebar.selectbox(
        "Select Secondary Variable",
        ['charges' if primary_var != 'charges' else 'age'] + 
        [var for var in ['age', 'bmi', 'children', 'region', 'smoker', 'sex'] if var != primary_var]
    )
    
    # Advanced Analysis Section
    st.header("📊 Advanced Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Relationship Analysis", "Categorical Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced distribution plot
            fig = plt.figure(figsize=(10, 6))
            if df[primary_var].dtype in ['int64', 'float64']:
                sns.histplot(data=df, x=primary_var, hue='smoker')
            else:
                sns.countplot(data=df, x=primary_var, hue='smoker')
            plt.title(f'Distribution of {primary_var} by Smoking Status')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Box plot with individual points
            fig = plt.figure(figsize=(10, 6))
            if df[primary_var].dtype in ['int64', 'float64']:
                sns.boxplot(data=df, y=primary_var, x='region', hue='smoker')
                plt.title(f'{primary_var} Distribution by Region')
            else:
                sns.boxplot(data=df, y='charges', x=primary_var, hue='smoker')
                plt.title(f'Charges Distribution by {primary_var}')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot without trend line
            if df[primary_var].dtype in ['int64', 'float64'] and df[secondary_var].dtype in ['int64', 'float64']:
                fig = px.scatter(df, x=primary_var, y=secondary_var, 
                               color='smoker',
                               title=f'Relationship between {primary_var} and {secondary_var}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Cannot create scatter plot for categorical variables")
        
        with col2:
            # Advanced violin plot
            fig = plt.figure(figsize=(10, 6))
            if df[primary_var].dtype in ['int64', 'float64']:
                sns.violinplot(data=df, y=primary_var, x='region', hue='smoker', split=True)
                plt.title(f'{primary_var} Distribution by Region and Smoking Status')
            else:
                sns.violinplot(data=df, y='charges', x=primary_var, hue='smoker', split=True)
                plt.title(f'Charges Distribution by {primary_var} and Smoking Status')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap for numerical variables
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            correlation = df[numerical_cols].corr()
            fig = px.imshow(correlation, 
                           title="Correlation Heatmap",
                           labels=dict(color="Correlation"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Advanced categorical analysis
            if df[primary_var].dtype not in ['int64', 'float64']:
                # Create joint plot for categorical variables
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x=primary_var, y='charges', hue='smoker')
                plt.title(f'Average Charges by {primary_var} and Smoking Status')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                # Alternative visualization for numerical primary variable
                fig = px.box(df, x='region', y=primary_var, color='smoker',
                           title=f'{primary_var} Distribution by Region and Smoking Status')
                st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Insights Section
    st.header("🔍 Detailed Insights")
    
    # Summary statistics
    if st.checkbox("Show Summary Statistics"):
        if df[primary_var].dtype in ['int64', 'float64']:
            summary = df.groupby('region')[primary_var].describe()
            st.write(f"Summary Statistics for {primary_var} by Region:")
            st.dataframe(summary)
        else:
            summary = df.groupby(primary_var)['charges'].describe()
            st.write(f"Charges Summary Statistics by {primary_var}:")
            st.dataframe(summary)
    
    # Cross-analysis
    if st.checkbox("Show Cross Analysis"):
        if df[primary_var].dtype not in ['int64', 'float64']:
            cross_analysis = pd.crosstab(df[primary_var], df['region'], margins=True)
            st.write(f"Cross Analysis of {primary_var} by Region:")
            st.dataframe(cross_analysis)
        else:
            grouped_analysis = df.groupby('region')[primary_var].agg(['mean', 'median', 'std', 'count'])
            st.write(f"Grouped Analysis of {primary_var} by Region:")
            st.dataframe(grouped_analysis)

if __name__ == "__main__":
    main()