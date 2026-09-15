import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

def plot_and_save_coefficients(importance_df, metric_label, is_log_scale=False, output_dir="outputs"):
    """
    Generates horizontal bar chart of top features 
    and saves file directly into outputs folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Isolate top features for visual balance
    if 'Coefficient' in importance_df.columns:
        top_pos = importance_df[importance_df['Coefficient'] > 0].head(10)
        top_neg = importance_df[importance_df['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True).head(10)
        plot_data = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient', ascending=False)
        x_col = 'Coefficient'
        title_str = 'Top Feature Coefficients (Lasso Pricing Weights)'
        
        # Adjust label to clarify if scale represents percentage multipliers or dollar impact
        x_label = 'Coefficient Value (% Impact on Pricing)' if is_log_scale else 'Coefficient Value ($ Impact on Pricing)'
        colors = ['#2b7bba' if val > 0 else '#d95f02' for val in plot_data[x_col]]
    else:
        # XGBoost importances are positive 
        plot_data = importance_df.head(20).copy()
        x_col = 'Importance'
        title_str = 'Top 20 XGBoost Feature Importances'
        x_label = 'Relative Feature Importance'
        colors = ['#2b7bba'] * len(plot_data)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=x_col, 
        y='Feature', 
        data=plot_data, 
        palette=colors,
        hue='Feature',
        legend=False
    )
    
    plt.title(title_str, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Engineered Features', fontsize=12)
    if x_col == 'Coefficient':
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    file_name = 'feature_coefficients.png' if x_col == 'Coefficient' else 'xgb_feature_importances.png'
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Visual asset successfully saved to: {file_path}")


def run_model_audit(pipeline, X_train, X_test, y_train, y_test):
    """
    Computes performance metrics, checks for overfitting, and 
    extracts final model weights/importances. 
    Supports LassoCV, XGBoost, GridSearchCV, and TransformedTargetRegressor wrappers.
    """
    # 1. Utilize GridSearchCV if hyperparameter tuning was used
    if isinstance(pipeline, GridSearchCV):
        best_estimator = pipeline.best_estimator_
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
    else:
        best_estimator = pipeline
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)

    print("=========================================")
    print("       MODEL PERFORMANCE AUDIT           ")
    print("=========================================")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score:  {test_score:.4f}")
    print(f"Generalization Delta: {abs(train_score - test_score):.4f}")
    
    pipeline_regressor = best_estimator.named_steps['regressor']
    preprocessor = best_estimator.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    
    # Track if the target was log-transformed
    is_log_scale = False
    
    # 2. Safely extract the inner core model if it is wrapped inside a target transformer
    if isinstance(pipeline_regressor, TransformedTargetRegressor):
        core_model = pipeline_regressor.regressor_
        is_log_scale = True
    else:
        core_model = pipeline_regressor
    
    # 3. Check model type and safely extract metrics from core_model
    if hasattr(core_model, 'coef_'):
        # --- LINEAR REGRESSION (LASSO) PATH ---
        coefficients = core_model.coef_
        chosen_alpha = getattr(core_model, 'alpha_', None)
        
        if chosen_alpha:
            print(f"Optimal Alpha Selected by CV: {chosen_alpha:.4f}")
        print(f"Total Dimensions Evaluated:   {len(coefficients)}")
        print(f"Dimensions Kept by Lasso:     {sum(coefficients != 0)}")
        print(f"Dimensions Dropped (Zeroed):  {sum(coefficients == 0)}")
        print("=========================================\n")
        
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        kept_features = feature_importance[feature_importance['Coefficient'] != 0].copy()
        kept_features['Abs_Value'] = kept_features['Coefficient'].abs()
        kept_features = kept_features.sort_values(by='Abs_Value', ascending=False)
        
        top_premiums = kept_features[kept_features['Coefficient'] > 0].head(5)
        top_discounts = kept_features[kept_features['Coefficient'] < 0].sort_values(by='Coefficient', ascending=True).head(5)
        
        print("--- TOP 5 ASSET PREMIUM DRIVERS ---")
        print(top_premiums[['Feature', 'Coefficient']].to_string(index=False))
        print("\n--- TOP 5 ASSET DISCOUNT DRIVERS ---")
        print(top_discounts[['Feature', 'Coefficient']].to_string(index=False))
        
        plot_and_save_coefficients(kept_features, metric_label='Coefficient', is_log_scale=is_log_scale)
        
    elif hasattr(core_model, 'feature_importances_'):
        # --- TREE-BASED ENSEMBLER (XGBOOST) PATH ---
        
        importances = core_model.feature_importances_
        print(f"Total Features Evaluated by XGB: {len(importances)}")
        print("=========================================\n")
        
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        print("--- TOP 10 XGBOOST PREDICTIVE DRIVERS ---")
        print(feature_importance.head(10).to_string(index=False))
        
        plot_and_save_coefficients(feature_importance, metric_label='Importance', is_log_scale=is_log_scale)
        
    print("\n-----------------------------------------")
