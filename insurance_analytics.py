import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D

def load_and_preprocess_data():
    """Load and preprocess the insurance dataset."""
    df = pd.read_csv('insurance.csv')
    categorical_features = ['sex', 'smoker', 'region']
    
    # One-hot encoding
    encode = OneHotEncoder(drop='first', sparse_output=False)
    df_encoded_columns = encode.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        df_encoded_columns, 
        columns=encode.get_feature_names_out(categorical_features)
    )
    
    # Combine encoded and numerical features
    enhanced_df = pd.concat(
        [df.drop(categorical_features, axis=1), encoded_df], 
        axis=1
    )
    
    # Handle infinite values
    enhanced_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df, enhanced_df

def main():
    st.title("Insurance Data Visualization Dashboard")
    
    # Load data
    df, enhanced_df = load_and_preprocess_data()
    
    # Age, BMI, Children vs Charges
    st.header("Relationships with Insurance Charges")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Age vs Charges
    ax1.scatter(enhanced_df['age'], enhanced_df['charges'], c='blue')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Charges')
    ax1.set_title('Age vs Charges')
    
    # BMI vs Charges
    ax2.scatter(enhanced_df['bmi'], enhanced_df['charges'], c='green')
    ax2.set_xlabel('BMI')
    ax2.set_ylabel('Charges')
    ax2.set_title('BMI vs Charges')
    
    # Children vs Charges
    sns.barplot(data=enhanced_df, x='children', y='charges', color='yellow', ax=ax3)
    ax3.set_title('Children vs Charges')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Smoker, Region, and BMI analysis
    st.header("Categorical Analysis")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Smoker vs Charges
    sns.boxplot(data=enhanced_df, x='smoker_yes', y='charges', palette='flare', ax=ax1)
    ax1.set_xlabel('Smoker')
    ax1.set_title('Smoker vs Charges')
    
    # Region vs Charges
    sns.barplot(data=df, x='region', y='charges', palette='mako', ax=ax2)
    ax2.set_xlabel('Region')
    ax2.set_title('Region vs Charges')
    
    # BMI vs Smoker with Charges
    scatter = ax3.scatter(enhanced_df['bmi'], enhanced_df['smoker_yes'], 
                         c=enhanced_df['charges'], s=enhanced_df['charges']/100)
    ax3.set_xlabel('BMI')
    ax3.set_ylabel('Smoker')
    ax3.set_title('BMI vs Smoker (with Charges)')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 3D Plot
    st.header("3D Analysis: Age vs BMI vs Charges")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(enhanced_df['age'], enhanced_df['bmi'], enhanced_df['charges'], 
                        s=50, alpha=0.7)
    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Charges')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.header("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = enhanced_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap of Numerical Features')
    st.pyplot(fig)
    
    # Age Distribution by Smoker Status and Charges Distribution
    st.header("Additional Distributions")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Age Distribution by Smoker Status
    sns.boxplot(data=enhanced_df, y='age', x='smoker_yes', ax=ax1)
    ax1.set_xlabel('Smoker')
    ax1.set_ylabel('Age')
    ax1.set_title('Age Distribution by Smoker Status')
    
    # Charges Distribution
    sns.histplot(data=enhanced_df, x='charges', kde=True, color='blue', bins=30, ax=ax2)
    ax2.set_xlabel('Charges')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Charges Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()