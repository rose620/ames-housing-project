#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_and_save_coefficients(kept_features, output_dir="outputs"):
    """
    Generates a clean horizontal bar chart of the top features 
    and saves the file directly into the outputs folder.
    """
    # Create the directory safely if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)
    
    # Isolate top 10 positive and top 10 negative features for visual balance
    top_pos = kept_features[kept_features['Coefficient'] > 0].head(10)
    top_neg = kept_features[kept_features['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True).head(10)
    plot_data = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient', ascending=False)
    
    # Initialize the plot layout
    plt.figure(figsize=(12, 8))
    
    # Create a divergent color palette (blue for premium, red for discounts)
    colors = ['#2b7bba' if coef > 0 else '#d95f02' for coef in plot_data['Coefficient']]
    
    sns.barplot(
        x='Coefficient', 
        y='Feature', 
        data=plot_data, 
        palette=colors,
        hue='Feature',
        legend=False
    )
    
    # Format labels and aesthetics for a professional look
    plt.title('Top Feature Coefficients (Lasso Pricing Weights)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Coefficient Value ($ Impact on Pricing)', fontsize=12)
    plt.ylabel('Engineered Features', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Baseline zero line
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot directly into the outputs folder
    file_path = os.path.join(output_dir, 'feature_coefficients.png')
    plt.savefig(file_path, dpi=300)
    plt.close() # Close plot to free up memory
    print(f"Visual asset successfully saved to: {file_path}")


def run_model_audit(pipeline, X_train, X_test, y_train, y_test):
    """
    Computes performance metrics, checks for overfitting, and 
    extracts final model coefficients for senior auditing.
    """
    # 1. Performance Validation
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print("=========================================")
    print("       MODEL PERFORMANCE AUDIT           ")
    print("=========================================")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score:  {test_score:.4f}")
    print(f"Generalization Delta: {abs(train_score - test_score):.4f}")
    
    # 2. Feature Selection Auditing
    lasso_model = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    
    chosen_alpha = lasso_model.alpha_
    coefficients = lasso_model.coef_
    feature_names = preprocessor.get_feature_names_out()
    
    print(f"Optimal Alpha Selected by CV: {chosen_alpha:.4f}")
    print(f"Total Dimensions Evaluated:   {len(coefficients)}")
    print(f"Dimensions Kept by Lasso:     {sum(coefficients != 0)}")
    print(f"Dimensions Dropped (Zeroed):  {sum(coefficients == 0)}")
    print("=========================================\n")
    
    # 3. Extract and Rank Weights
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    kept_features = feature_importance[feature_importance['Coefficient'] != 0].copy()
    kept_features['Abs_Coefficient'] = kept_features['Coefficient'].abs()
    kept_features = kept_features.sort_values(by='Abs_Coefficient', ascending=False)
    
    # Separate text metrics printout
    top_premiums = kept_features[kept_features['Coefficient'] > 0].head(5)
    top_discounts = kept_features[kept_features['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True).head(5)
    
    print("--- TOP 5 ASSET PREMIUM DRIVERS ---")
    print(top_premiums[['Feature', 'Coefficient']].to_string(index=False))
    print("\n--- TOP 5 ASSET DISCOUNT DRIVERS ---")
    print(top_discounts[['Feature', 'Coefficient']].to_string(index=False))
    print("\n-----------------------------------------")
    
    # 4. Trigger Automatic Visualization Saving
    plot_and_save_coefficients(kept_features)

