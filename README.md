# Breast-Cancer-Classification-using-Machine-Learning
This project implements and compares multiple ML models to classify breast cancer tumors as benign or malignant using the Wisconsin Breast Cancer Dataset. The goal is to demonstrate how classical and advanced ML models can be applied to biomedical data for accurate disease prediction.

The pipeline includes:
* Data preprocessing and cleaning
* Feature scaling and encoding
* Training and evaluation of multiple ML models
* Model comparison and interpretation
* Model serialization for reuse

# Dataset
Source: Wisconsin Breast Cancer Dataset (UCI ML Repository)
Features: Cytological characteristics extracted from fine needle aspirates (FNA), including:

* Clump Thickness
* Uniformity of Cell Size
* Uniformity of Cell Shape
* Marginal Adhesion
* Single Epithelial Cell Size
* Bare Nuclei
* Bland Chromatin
* Normal Nucleoli
* Mitosis

Target Variable:
* 2 -> Benign (encoded as 0)
* 4 -> Malignant (encoded as 1)

# Preprocessing 
* Dropped non-informative id column
* Handled missing values (?) using median imputation
* Converted labels to binary format
* Applied feature scaling (StandardScaler) for scale-sensitive models
* Used stratified train-test split to handle class imbalance

# Models Implemented
The following models were trained and evaluated on the same dataset and split:

1. Logistic Regression (Baseline)
* Linear classifier
* Interpretable coefficients
* Used as a baseline model

2. Support Vector Machine (RBF Kernel)
* Captures non-linear decision boundaries
* Effective for small-to-medium biomedical datasets

3. Random Forest
* Ensemble tree-based model
* Provides feature importance for biological interpretation

4. XGBoost
* Gradient boosting framework
* Captures complex feature interactions
* Best overall performance among tree-based models

# Model Performance 

| Model | Accuracy |
| ----- | -------- |
| Logistic Regression | 1.00 |
| SVM | 0.97 |
| Random Forest | 0.94 |
| XGBoost | 0.97 |

Logistic Regression achieved perfect accuracy on the held-out test set, while non-linear models (SVM, XGBoost) also performed strongly, indicating that both linear and non-linear relationships exist in cytological features.

# Feature Importance (Random Forest)
Random Forest highlights biologically relevant predictors:
* Uniformity of Cell Size
* Bare Nuclei
* Uniformity of Cell Shape
* Bland Chromatin

These features are well-aligned with known pathological indicators of malignancy.

# Prediction Example
```bash
sample = np.array([[5,10,10,10,7,7,3,8,9]])
prediction = model.predict(sample)[0]
```
Output:
```bash
Prediction: Malignant
```
# Model Persistence

All trained models are serialized using pickle for future inference:
* LogisticRegression.m
* SVM.m
* RandomForest.m
* XGBoost.m

# Technologies Used
* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Pickle

# Bioinformatics Relevance

This project demonstrates how machine learning can assist in computer-aided diagnosis by learning discriminative patterns from cytological data. The comparison of linear and non-linear models highlights the importance of model selection when analyzing biological features with complex dependencies.

# Future Improvements
* ROC-AUC and precision–recall analysis
* Cross-validation for robustness
* Hyperparameter optimization
* SHAP-based explainability for XGBoost
* Deployment as a simple web application
