# Machine Learning Loan Payback Prediction

##  Description
A machine learning project that predicts whether a customer will pay back their loan or default. This binary classification model helps financial institutions make better lending decisions.

##  Project Objective
Predict loan repayment status (Will Pay at which plobability / Will not pay at which plobability) based on customer features

##  Dataset
- **Source**: [/kaggle/input/loan-pay-back-dataset-2025-2026-credit-to-kaggle/train.csv]
- **Features**: [13] features including id	annual_income	debt_to_income_ratio	credit_score	loan_amount	interest_rate	gender	marital_status	education_level	employment_status	loan_purpose	grade_subgrade	loan_paid_back
- **Target**: Loan_pai_back

##  Technologies Used
- Python 3.x
- Pandas & NumPy - Data manipulation
- Scikit-learn - Machine learning
- Matplotlib & Seaborn - Data visualization
- Jupyter Notebook - Development environment

##  Project Structure
```
├── loan_prediction.ipynb      # Main analysis notebook
├── model.pkl                  # Trained model
├── scaler.pkl                 # Feature scaler
├── label_encoders.pkl         # Categorical encoders
├── feature_names.pkl          # Feature names list
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

##  Models Implemented
- Logistic Regression
- Random Forest 
- XGBoost 

##  Results
- **Best Model**: [XGBoost]
- **Accuracy**: [0.90]%


##  How to Run
1. Clone this repository:
```bash
   git clone https://github.com/Umuhoza90709/Machine_learning_Loan_payback_prediction.git
```

2. Install required packages:
```bash
   pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
   jupyter notebook loan_prediction.ipynb
```

4. Run all cells to see the analysis

##  Model Deployment Files
The trained model and preprocessing objects are saved as:
- `model.pkl` - Trained classification model
- `scaler.pkl` - StandardScaler for feature scaling
- `label_encoders.pkl` - Encoders for categorical variables
- `feature_names.pkl` - List of feature names used in training

##  Links
- **Kaggle Notebook**: [Add your Kaggle notebook link]
- **Dataset**: [/kaggle/input/loan-pay-back-dataset-2025-2026-credit-to-kaggle/train.csv]

##  Author
**Umuhoza90709**
- GitHub: [@Umuhoza90709](https://github.com/Umuhoza90709)

##  License
This project is licensed under the MIT License.
