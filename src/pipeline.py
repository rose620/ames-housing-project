#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src/pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV

def create_production_pipeline(numeric_cols, categorical_cols):
    """
    Creates a robust, non-leaking ColumnTransformer and couples it 
    directly with an optimized LassoCV regressor within a single Pipeline.
    """
    
    # 1. Numerical Engineering: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Engineering: Impute with mode, then safely one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine Preprocessors across explicit column subsets
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # 4. Bind the Preprocessor and Cross-Validated Lasso Model together
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LassoCV(cv=5, random_state=42, max_iter=50000))
    ])
    
    return full_pipeline

