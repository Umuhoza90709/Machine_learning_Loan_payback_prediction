import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# LOAD TRAINED MODEL AND PREPROCESSING OBJECTS


print("Loading model and preprocessing objects...")

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print(" All files loaded successfully!")
    print(f" Expected features: {feature_names}")
except Exception as e:
    print(f" Error loading files: {e}")
    print("Make sure you have run the training script first!")
    exit()


# PREDICTION FUNCTION 


def predict_loan_approval(
    annual_income,
    debt_to_income_ratio,
    credit_score,
    loan_amount,
    interest_rate,
    gender,
    marital_status,
    education_level,
    employment_status,
    loan_purpose,
    grade_subgrade
):
    """
    Predict loan approval with all 11 features
    
    Parameters:
    -----------
    annual_income : float
        Annual income in dollars
    debt_to_income_ratio : float
        Debt to income ratio (e.g., 0.35 for 35%)
    credit_score : int
        Credit score (300-850)
    loan_amount : float
        Requested loan amount in dollars
    interest_rate : float
        Interest rate percentage (e.g., 10.5)
    gender : str
        Gender ('Male', 'Female', 'Other')
    marital_status : str
        Marital status ('Single', 'Married', 'Divorced', 'Widowed')
    education_level : str
        Education level ('High School', 'Bachelor', 'Master', 'PhD')
    employment_status : str
        Employment status ('Employed', 'Self-Employed', 'Unemployed')
    loan_purpose : str
        Purpose of loan ('Personal', 'Education', 'Home', 'Auto', 'Business')
    grade_subgrade : str
        Loan grade (e.g., 'A1', 'A2', 'B1', 'B2', 'C1', etc.)
    
    Returns:
    --------
    dict : Prediction result with probability
    """
    
    # Create DataFrame with all 11 features
    test_data = pd.DataFrame({
        'annual_income': [annual_income],
        'debt_to_income_ratio': [debt_to_income_ratio],
        'credit_score': [credit_score],
        'loan_amount': [loan_amount],
        'interest_rate': [interest_rate],
        'gender': [gender],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'employment_status': [employment_status],
        'loan_purpose': [loan_purpose],
        'grade_subgrade': [grade_subgrade]
    })
    
    # Encode categorical features using saved label encoders
    categorical_features = ['gender', 'marital_status', 'education_level', 
                           'employment_status', 'loan_purpose', 'grade_subgrade']
    
    for col in categorical_features:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                test_data[col] = le.transform(test_data[col])
            except ValueError as e:
                print(f" Warning: Unknown value for {col}: {test_data[col].values[0]}")
                print(f"   Known values: {list(le.classes_)}")
                # Use the most common class as fallback
                test_data[col] = 0
    
    # Ensure column order matches training data
    test_data = test_data[feature_names]
    
    # Scale features
    test_data_scaled = scaler.transform(test_data)
    
    # Make prediction
    prediction = model.predict(test_data_scaled)[0]
    prediction_proba = model.predict_proba(test_data_scaled)[0]
    
    result = {
        'prediction': 'APPROVED ' if prediction == 1 else 'REJECTED ',
        'prediction_value': int(prediction),
        'approval_probability': float(prediction_proba[1]),
        'rejection_probability': float(prediction_proba[0])
    }
    
    return result

# ========================================
# EXAMPLE USAGE - TEST CASES
# ========================================

print("\n" + "="*60)
print("LOAN PREDICTION EXAMPLES")
print("="*60)

# Example 1: High-quality applicant (should be APPROVED)
print("\n Example 1: High-Quality Applicant")
print("-" * 60)
result1 = predict_loan_approval(
    annual_income=85000,
    debt_to_income_ratio=0.25,
    credit_score=750,
    loan_amount=15000,
    interest_rate=8.5,
    gender='Male',
    marital_status='Married',
    education_level='Bachelor',
    employment_status='Employed',
    loan_purpose='Home',
    grade_subgrade='A1'
)

print(f"Prediction: {result1['prediction']}")
print(f"Approval Probability: {result1['approval_probability']:.2%}")
print(f"Rejection Probability: {result1['rejection_probability']:.2%}")

# Example 2: Risky applicant (should be REJECTED)
print("\n Example 2: High-Risk Applicant")
print("-" * 60)
result2 = predict_loan_approval(
    annual_income=30000,
    debt_to_income_ratio=0.65,
    credit_score=580,
    loan_amount=25000,
    interest_rate=18.5,
    gender='Female',
    marital_status='Single',
    education_level='High School',
    employment_status='Unemployed',
    loan_purpose='Personal',
    grade_subgrade='D2'
)

print(f"Prediction: {result2['prediction']}")
print(f"Approval Probability: {result2['approval_probability']:.2%}")
print(f"Rejection Probability: {result2['rejection_probability']:.2%}")

# Example 3: Medium-quality applicant
print("\n Example 3: Medium-Quality Applicant")
print("-" * 60)
result3 = predict_loan_approval(
    annual_income=55000,
    debt_to_income_ratio=0.40,
    credit_score=680,
    loan_amount=18000,
    interest_rate=12.0,
    gender='Female',
    marital_status='Married',
    education_level='Master',
    employment_status='Self-Employed',
    loan_purpose='Education',
    grade_subgrade='B2'
)

print(f"Prediction: {result3['prediction']}")
print(f"Approval Probability: {result3['approval_probability']:.2%}")
print(f"Rejection Probability: {result3['rejection_probability']:.2%}")


# INTERACTIVE PREDICTION


print("\n" + "="*60)
print("INTERACTIVE LOAN PREDICTION")
print("="*60)
print("\nEnter applicant details for prediction:\n")

def get_user_input():
    """Get user input for all 11 features"""
    
    try:
        # Numeric features
        annual_income = float(input("Annual Income ($): "))
        debt_to_income_ratio = float(input("Debt-to-Income Ratio (0-1, e.g., 0.35): "))
        credit_score = int(input("Credit Score (300-850): "))
        loan_amount = float(input("Loan Amount ($): "))
        interest_rate = float(input("Interest Rate (%, e.g., 10.5): "))
        
        # Categorical features
        print("\nGender options: Male, Female, Other")
        gender = input("Gender: ").strip()
        
        print("\nMarital Status options: Single, Married, Divorced, Widowed")
        marital_status = input("Marital Status: ").strip()
        
        print("\nEducation Level options: High School, Bachelor, Master, PhD")
        education_level = input("Education Level: ").strip()
        
        print("\nEmployment Status options: Employed, Self-Employed, Unemployed")
        employment_status = input("Employment Status: ").strip()
        
        print("\nLoan Purpose options: Personal, Education, Home, Auto, Business")
        loan_purpose = input("Loan Purpose: ").strip()
        
        print("\nGrade Subgrade examples: A1, A2, B1, B2, C1, C2, D1, D2, E1, E2")
        grade_subgrade = input("Grade Subgrade: ").strip()
        
        # Make prediction
        result = predict_loan_approval(
            annual_income=annual_income,
            debt_to_income_ratio=debt_to_income_ratio,
            credit_score=credit_score,
            loan_amount=loan_amount,
            interest_rate=interest_rate,
            gender=gender,
            marital_status=marital_status,
            education_level=education_level,
            employment_status=employment_status,
            loan_purpose=loan_purpose,
            grade_subgrade=grade_subgrade
        )
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\n Prediction: {result['prediction']}")
        print(f" Approval Probability: {result['approval_probability']:.2%}")
        print(f" Rejection Probability: {result['rejection_probability']:.2%}")
        print("="*60)
        
    except ValueError as e:
        print(f"\n Invalid input: {e}")
        print("Please enter valid numeric values.")
    except Exception as e:
        print(f"\n Error: {e}")

# Uncomment to enable interactive mode
# get_user_input()


# BATCH PREDICTION FROM CSV


def batch_predict(csv_file):
    """
    Predict loan approvals for multiple applicants from CSV
    
    CSV should contain columns for all 11 features:
    - annual_income
    - debt_to_income_ratio
    - credit_score
    - loan_amount
    - interest_rate
    - gender
    - marital_status
    - education_level
    - employment_status
    - loan_purpose
    - grade_subgrade
    """
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f" Loaded {len(df)} applicants from {csv_file}")
        
        # Make predictions
        predictions = []
        probabilities = []
        
        for idx, row in df.iterrows():
            result = predict_loan_approval(
                annual_income=row['annual_income'],
                debt_to_income_ratio=row['debt_to_income_ratio'],
                credit_score=row['credit_score'],
                loan_amount=row['loan_amount'],
                interest_rate=row['interest_rate'],
                gender=row['gender'],
                marital_status=row['marital_status'],
                education_level=row['education_level'],
                employment_status=row['employment_status'],
                loan_purpose=row['loan_purpose'],
                grade_subgrade=row['grade_subgrade']
            )
            predictions.append(result['prediction'])
            probabilities.append(result['approval_probability'])
        
        # Add results to dataframe
        df['prediction'] = predictions
        df['approval_probability'] = probabilities
        
        # Save results
        output_file = csv_file.replace('.csv', '_predictions.csv')
        df.to_csv(output_file, index=False)
        print(f" Predictions saved to {output_file}")
        
        # Summary
        approved = sum(1 for p in predictions if 'APPROVED' in p)
        rejected = len(predictions) - approved
        
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total Applicants: {len(df)}")
        print(f"Approved: {approved} ({approved/len(df)*100:.1f}%)")
        print(f"Rejected: {rejected} ({rejected/len(df)*100:.1f}%)")
        print("="*60)
        
        return df
        
    except FileNotFoundError:
        print(f" File not found: {csv_file}")
    except KeyError as e:
        print(f" Missing required column: {e}")
        print(f"Required columns: {feature_names}")
    except Exception as e:
        print(f" Error: {e}")

# Example: Uncomment to predict from CSV
# batch_predict('new_applicants.csv')

print("\n Prediction system ready!")
print("You can now use predict_loan_approval() function with all 11 features.")