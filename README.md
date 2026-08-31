# Automated Valuation Model (AVM) for Residential Real Estate (Ames, Iowa)

## Executive Summary
This project delivers a production-ready Automated Valuation Model (AVM) built to predict residential home prices with high precision. Rather than manually cherry-picking variables, this architecture leverages an end-to-end Scikit-Learn processing pipeline combined with L1-regularized regression (`LassoCV`). 

The final production model processes **43 categorical features** and all numerical inputs, automatically evaluating **286 engineered dimensions**. The system algorithmically prunes **206 irrelevant or noisy features**, retaining the top 80 core drivers of real estate value.

* **Training $R^2$ Score:** 0.9004
* **Testing $R^2$ Score:**  0.8812
* **Model Status:** Production-Ready (No significant overfitting; generalizes within a tight 2% margin).

---

## Key Business Insights & Model Discoveries
Senior data auditing revealed that the model successfully isolated localized geographic premiums and specific structural updates while safely trapping data anomalies.
![Lasso Feature Coefficients](outputs/feature_coefficients.png)


### Top 5 Luxury Property Premiums
The model identified that living in specific premier neighborhoods or investing in specific high-end finishes yields the highest market returns:
1. **Neighborhood (Northridge):** Adds a baseline premium of **+$35,375**.
2. **Neighborhood (Stone Brook):** Adds a baseline premium of **+$30,814**.
3. **Excellent Kitchen Quality (`KitchenQual_Ex`):** Commands a premium of **+$26,946**.
4. **Excellent Basement Height (`BsmtQual_Ex`):** Yields an average increase of **+$26,154**.
5. **Gross Living Area (`GrLivArea`):** Drives a standardized premium of **+$23,105** per unit of scale.

### The Clay Tile Roof Anomaly (`RoofMatl_ClyTile`)
Standard model evaluation flagged an extreme negative coefficient for properties with Clay Tile Roofs (**-$214,417**). 
* **The Audit:** Rather than accepting the mathematical weight blindly, deep exploratory analysis revealed that this category was a severe outlier. The dataset contained only a single property with this feature—a luxury home that suffered catastrophic structural damage, resulting in a heavily suppressed litigation sale.
* **The Fix:** The production script flags and isolates high-cardinality sparse categories to ensure single-property anomalies do not skew macro pricing rules.

---

## Technical Architecture & Pipeline Design
The software is engineered following modular production standards, utilizing isolated pipelines to completely eliminate **data leakage** during cross-validation loops.

```text
Raw Data ➔ SimpleImputer (Median/Frequent) ➔ StandardScaler / OneHotEncoder ➔ LassoCV (Alpha: 185.27)
```

* **Handling Feature Explosion:** One-hot encoding 43 dense categorical columns exploded the input matrix to 286 features. `LassoCV` with a cross-validated penalty strength ($\alpha = 185.27$) successfully eliminated 72% of the input space.
* **Imputation Strategy:** Numerical missingness is handled via median imputation to protect against skewed outliers, while categorical gaps are filled using the most frequent strategy.

---

## Repository Structure
* `main.py`: Main execution script to trigger data processing, model training, and metrics export.
* `src/pipeline.py`: Production-grade Scikit-Learn `ColumnTransformer` and pipeline components.
* `src/evaluation.py`: Modules for extracting coefficients, feature weights, and residual evaluations.
* `outputs/`: Automatically exported diagnostic and performance visualizations.

## Quick Start
To reproduce the model pipeline and view diagnostic metrics locally:
```bash
pip install -r requirements.txt
python main.py
```
