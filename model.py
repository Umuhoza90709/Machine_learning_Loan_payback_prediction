import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from pathlib import Path

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
        # Paths relative to backend directory
        self.model_path = str(Path(__file__).parent / 'model.pkl')
        self.scaler_path = str(Path(__file__).parent / 'scaler.pkl')
        self.encoders_path = str(Path(__file__).parent / 'label_encoders.pkl')
        self.features_path = str(Path(__file__).parent / 'feature_names.pkl')
        
        self._load_model()
        self._load_scaler()
        self._load_encoders()
        self._load_feature_names()
    
    def _load_model(self):
        """Load the model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(" Model loaded successfully")
            else:
                print(f" Model file not found at {self.model_path}")
        except Exception as e:
            print(f" Error loading model: {str(e)}")
            self.model = None

    def _load_scaler(self):
        """Load the scaler from disk"""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(" Scaler loaded successfully")
            else:
                print(f" Scaler file not found at {self.scaler_path}")
        except Exception as e:
            print(f" Error loading scaler: {str(e)}")
            self.scaler = None
    
    def _load_encoders(self):
        """Load label encoders from disk"""
        try:
            if os.path.exists(self.encoders_path):
                self.label_encoders = joblib.load(self.encoders_path)
                print(" Label encoders loaded successfully")
                
                # Print what values each encoder expects
                print("\n Expected values for categorical features:")
                for key, encoder in self.label_encoders.items():
                    if hasattr(encoder, 'classes_'):
                        print(f"  {key}: {list(encoder.classes_)}")
            else:
                print(f" Label encoders file not found at {self.encoders_path}")
        except Exception as e:
            print(f" Error loading encoders: {str(e)}")
            self.label_encoders = {}
    
    def _load_feature_names(self):
        """Load feature names from disk"""
        try:
            if os.path.exists(self.features_path):
                self.feature_names = joblib.load(self.features_path)
                print(f" Feature names loaded: {self.feature_names}")
            else:
                print(f" Feature names file not found, using default order")
                # Default feature order matching your training
                self.feature_names = [
                    'annual_income', 'debt_to_income_ratio', 'credit_score', 
                    'loan_amount', 'interest_rate', 'gender', 'marital_status', 
                    'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade'
                ]
        except Exception as e:
            print(f" Error loading feature names: {str(e)}")
            self.feature_names = []
    
    def _normalize_categorical_value(self, column, value):
        """Normalize categorical values to match encoder expectations"""
        if not value or not isinstance(value, str):
            return value
        
        # Get expected values from encoder
        if column in self.label_encoders and hasattr(self.label_encoders[column], 'classes_'):
            expected_values = list(self.label_encoders[column].classes_)
            
            # Try exact match first
            if value in expected_values:
                return value
            
            # Try case-insensitive match
            value_lower = value.lower().strip()
            for expected in expected_values:
                if expected.lower() == value_lower:
                    return expected
            
            # Try partial match
            for expected in expected_values:
                if expected.lower().replace(' ', '').replace('_', '') == value_lower.replace(' ', '').replace('_', ''):
                    return expected
            
            # If still no match, return first class as default
            print(f" Value '{value}' not found for {column}, using default: {expected_values[0]}")
            return expected_values[0]
        
        return value
    
    def predict_single(self, features):
        """
        Make prediction for a single loan application from form data.
        
        Args:
            features: Dictionary containing loan application features from the form
            
        Returns:
            tuple: (prediction, probability) 
                   prediction: 1 for APPROVED, 0 for REJECTED
                   probability: float between 0 and 1 (confidence)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded. Please check if model.pkl and scaler.pkl exist.")
        
        if not self.feature_names:
            raise ValueError("Feature names not loaded. Please check if feature_names.pkl exists.")
            
        try:
            # Categorical columns that need encoding
            categorical_cols = ['gender', 'marital_status', 'education_level', 
                              'employment_status', 'loan_purpose', 'grade_subgrade']
            
            # Encode categorical features
            encoded_features = features.copy()
            for col in categorical_cols:
                if col in encoded_features and col in self.label_encoders:
                    try:
                        # Normalize the value before encoding
                        normalized_value = self._normalize_categorical_value(col, encoded_features[col])
                        encoded_features[col] = self.label_encoders[col].transform([normalized_value])[0]
                    except ValueError as e:
                        # Get expected values for better error message
                        expected = list(self.label_encoders[col].classes_) if hasattr(self.label_encoders[col], 'classes_') else 'unknown'
                        raise ValueError(f"Invalid value for {col}: '{encoded_features[col]}'. Expected one of: {expected}")
                elif col not in encoded_features:
                    # Use first class as default if feature is missing
                    if col in self.label_encoders and hasattr(self.label_encoders[col], 'classes_'):
                        encoded_features[col] = 0  # First encoded value
            
            # Create DataFrame with exact feature names from training
            features_df = pd.DataFrame([[encoded_features.get(col, 0) for col in self.feature_names]], 
                                      columns=self.feature_names)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Get ML model prediction
            prediction = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Get probability for approval (class 1)
            if len(proba) == 1:
                probability = proba[0] if prediction == 1 else (1 - proba[0])
            else:
                probability = proba[1]  # Probability of class 1 (approved)
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def predict_batch(self, features_list):
        """
        Make predictions for multiple loan applications.
        
        Args:
            features_list: List of dictionaries containing loan application features
            
        Returns:
            list: List of tuples (prediction, probability) for each application
        """
        results = []
        for i, features in enumerate(features_list):
            try:
                prediction, probability = self.predict_single(features)
                results.append((prediction, probability))
            except Exception as e:
                print(f" Error predicting application {i+1}: {str(e)}")
                results.append((None, None))
        
        return results