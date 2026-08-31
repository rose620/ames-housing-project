#!/usr/bin/env python
# coding: utf-8

# In[3]:


# main.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Import your custom modules
from src.pipeline import create_production_pipeline
from src.evaluation import run_model_audit

def main():
    # 1. Load your dataset (Replace with your actual path or URL fetch)
    data_path = os.path.join('data', 'train.csv')
    df = pd.read_csv(data_path)

    # For demonstration, assuming 'df' is pre-loaded
    print("Loading and preparing Ames Housing dataset...")
    
    # 2. Isolate Target and Features
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # 3. Programmatically separate data pools (Chained solution!)
    numeric_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 4. Establish Train/Test Isolation Splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Initialize Production Pipeline
    model_pipeline = create_production_pipeline(numeric_cols, categorical_cols)
    
    # 6. Fit entire pipeline natively without data leakage
    print("Executing preprocessing steps and fitting LassoCV model...")
    model_pipeline.fit(X_train, y_train)
    
    # 7. Execute Senior Model Audit
    run_model_audit(model_pipeline, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()


# In[ ]:




