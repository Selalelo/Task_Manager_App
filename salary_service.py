# salary_service.py
import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalaryDataCollector:
    """Phase 1: Data Collection and Quality Assessment"""
    
    def __init__(self, api_base_url=None, api_key=None, host=None):
        self.api_base_url = api_base_url or os.getenv("SALARY_API_URL")
        self.api_key = api_key or os.getenv("SALARY_API_KEY")
        self.host = host or os.getenv("HOST")
        self.quality_threshold = {"min_sample_size": 5, "required_confidence": ["MEDIUM", "HIGH", "VERY_HIGH"]}
        
    def make_api_call(self, job_title, location, experience_level, location_type="CITY"):
        """Make API call with error handling"""
        try:
            params = {
                "job_title": job_title,
                "location": location,
                "location_type": location_type,
                "years_of_experience": experience_level
            }
            
            headers = {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.host
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(self.api_base_url, params=params, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed for {job_title}, {location}, {experience_level}: {e}")
            return None

    def assess_data_quality(self, api_response):
        """Assess if API response meets quality standards"""
        if not api_response or api_response.get("status") != "OK":
            return False, "API call failed"
        
        data = api_response.get("data", [])
        if not data:
            return False, "No data returned"
        
        sample_data = data[0]  # Assuming single response
        sample_size = sample_data.get("salary_count", 0)
        confidence = sample_data.get("confidence", "LOW")
        
        quality_score = {
            "sample_size": sample_size,
            "confidence": confidence,
            "meets_threshold": (
                sample_size >= self.quality_threshold["min_sample_size"] and 
                confidence in self.quality_threshold["required_confidence"]
            )
        }
        
        return quality_score["meets_threshold"], quality_score
    
    def collect_strategic_data(self):
        """Collect data for strategic job/location/experience combinations"""
        
        # Define strategic combinations based on common HR needs
        priority_combinations = [
            # Data science roles
            ("data scientist", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "johannesburg", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "ONE_TO_THREE"),
            ("data scientist", "johannesburg", "ONE_TO_THREE"),
            ("data scientist", "cape town", "FOUR_TO_SIX"),
            ("data scientist", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "SEVEN_TO_NINE"),
            ("data scientist", "johannesburg", "SEVEN_TO_NINE"),

            # Data analyst roles
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data analyst", "johannesburg", "LESS_THAN_ONE"),
            ("data analyst", "cape town", "ONE_TO_THREE"),
            ("data analyst", "johannesburg", "ONE_TO_THREE"),
            ("data analyst", "cape town", "FOUR_TO_SIX"),
            ("data analyst", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "SEVEN_TO_NINE"),
            ("data analyst", "johannesburg", "SEVEN_TO_NINE"),

            # Software roles
            ("software developer", "cape town", "LESS_THAN_ONE"),
            ("software developer", "johannesburg", "LESS_THAN_ONE"),
            ("software engineer", "cape town", "ONE_TO_THREE"),
            ("software engineer", "johannesburg", "ONE_TO_THREE"),
            ("software engineer", "cape town", "FOUR_TO_SIX"),
            ("software engineer", "johannesburg", "FOUR_TO_SIX"),
            ("software engineer", "cape town", "SEVEN_TO_NINE"),
            ("software engineer", "johannesburg", "SEVEN_TO_NINE"),
            ("full stack developer", "johannesburg", "ONE_TO_THREE"),
            ("backend developer", "johannesburg", "FOUR_TO_SIX"),
            
            # Other tech roles
            ("devops engineer", "cape town", "FOUR_TO_SIX"),
            ("machine learning engineer", "johannesburg", "FOUR_TO_SIX"),
            ("product manager", "cape town", "FOUR_TO_SIX"),
            ("ui/ux designer", "cape town", "ONE_TO_THREE"),
        ]
        
        collected_data = []
        quality_report = {}
        
        logger.info(f"Starting data collection for {len(priority_combinations)} combinations...")
        
        for i, (job_title, location, experience) in enumerate(priority_combinations):
            logger.info(f"Collecting data {i+1}/{len(priority_combinations)}: {job_title} in {location}")
            
            # Make API call
            api_response = self.make_api_call(job_title, location, experience)
            
            if api_response:
                # Assess quality
                is_quality, quality_info = self.assess_data_quality(api_response)
                quality_report[(job_title, location, experience)] = quality_info
                
                if is_quality:
                    # Extract and clean data
                    data_point = self.extract_data_point(api_response, job_title, location, experience)
                    if data_point:
                        collected_data.append(data_point)
                        logger.info(f"✓ High quality data collected: {quality_info['sample_size']} samples, {quality_info['confidence']} confidence")
                else:
                    logger.warning(f"✗ Low quality data: {quality_info}")
            
            # Respect API rate limits
            time.sleep(1)
        
        logger.info(f"Data collection complete: {len(collected_data)} high-quality data points collected")
        return collected_data, quality_report
    
    def extract_data_point(self, api_response, job_title, location, experience):
        """Extract clean data point from API response"""
        try:
            data = api_response["data"][0]
            
            return {
                "job_title": job_title,
                "location": location,  
                "experience_level": experience,
                "min_salary": data.get("min_salary"),
                "max_salary": data.get("max_salary"),
                "median_salary": data.get("median_salary"),
                "min_base_salary": data.get("min_base_salary"),
                "max_base_salary": data.get("max_base_salary"), 
                "median_base_salary": data.get("median_base_salary"),
                "salary_period": data.get("salary_period", "MONTH"),
                "salary_currency": data.get("salary_currency"),
                "salary_count": data.get("salary_count"),
                "confidence": data.get("confidence"),
                "collected_at": datetime.now().isoformat()
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting data point: {e}")
            return None

class SalaryPredictor:
    """Phase 1: Baseline ML Model"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = ['job_title', 'location', 'experience_level']
        self.model_path = "salary_model.joblib"
        
    def prepare_features(self, df):
        """Encode categorical features for ML model"""
        df_encoded = df.copy()
        
        for column in self.feature_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                # Handle unseen categories during prediction
                le = self.label_encoders[column]
                mask = df[column].isin(le.classes_)
                df_encoded[f'{column}_encoded'] = -1  # Default for unseen categories
                df_encoded.loc[mask, f'{column}_encoded'] = le.transform(df.loc[mask, column])
        
        return df_encoded
    
    def train_model(self, training_data):
        """Train the baseline RandomForest model"""
        df = pd.DataFrame(training_data)
        
        if len(df) < 10:
            raise ValueError("Insufficient training data. Need at least 10 data points.")
        
        # Normalize salary period (convert monthly to annual)
        df['annual_median_salary'] = df.apply(lambda row: 
            row['median_salary'] * 12 if row['salary_period'] == 'MONTH' 
            else row['median_salary'], axis=1)
        
        # Prepare features
        df_encoded = self.prepare_features(df)
        
        # Features and target
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        X = df_encoded[feature_cols]
        y = df_encoded['annual_median_salary']
        
        if len(df) < 30:
            # Don't split - use all data for training and testing
            X_train = X_test = X
            y_train = y_test = y
            logger.warning("Small dataset: Using all data for both training and testing")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=2
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        evaluation = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "training_samples": len(df),
            "test_samples": len(y_test)
        }
        
        self.is_trained = True
        logger.info(f"Model trained successfully: R² = {evaluation['r2']:.3f}, MAE = {evaluation['mae']:.0f}")
        
        # Save the model
        self.save_model(self.model_path)
        
        return evaluation
    
    def predict_salary(self, job_title, location, experience_level):
        """Predict salary for given inputs"""
        if not self.is_trained:
            # Try to load the model if it exists
            try:
                self.load_model(self.model_path)
            except:
                raise ValueError("Model must be trained before making predictions")
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'job_title': job_title,
            'location': location,
            'experience_level': experience_level
        }])
        
        # Encode features
        input_encoded = self.prepare_features(input_data)
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        X_input = input_encoded[feature_cols]
        
        # Make prediction
        prediction = self.model.predict(X_input)[0]
        
        # Estimate range (±20% of prediction)
        prediction_range = (prediction * 0.8, prediction * 1.2)
        
        return {
            "predicted_annual_salary": round(prediction),
            "predicted_range": (round(prediction_range[0]), round(prediction_range[1])),
            "source": "ML_MODEL"
        }
    
    def save_model(self, filepath):
        """Save trained model and encoders"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and encoders"""
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.label_encoders = model_data["label_encoders"]
        self.feature_columns = model_data["feature_columns"]
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")

class CompetitivenessAnalyzer:
    """Phase 1: Basic Competitiveness Analysis"""
    
    @staticmethod
    def analyze_competitiveness(predicted_salary, predicted_range, user_budget, confidence="MEDIUM"):
        """Analyze how competitive user's budget is vs market prediction"""
        
        min_market, max_market = predicted_range
        pct_vs_median = (user_budget - predicted_salary) / predicted_salary * 100
        
        # Determine competitiveness level
        if user_budget > max_market:
            level = "highly_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% above market - Highly Competitive"
            color = "green"
            recommendation = "Excellent positioning to attract top talent"
            
        elif user_budget >= predicted_salary:
            level = "competitive" 
            flag = f"Budget is {abs(pct_vs_median):.1f}% above median - Competitive"
            color = "blue"
            recommendation = "Good positioning for quality candidates"
            
        elif user_budget >= min_market:
            level = "somewhat_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below median but within market range"
            color = "yellow"
            recommendation = "May attract junior or motivated candidates"
            
        else:
            level = "not_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below market - Not Competitive" 
            color = "red"
            recommendation = "Consider increasing budget or adjusting requirements"
        
        return {
            "competitiveness_level": level,
            "flag": flag,
            "color": color,
            "percentage_difference": round(pct_vs_median, 1),
            "market_position": {
                "user_budget": user_budget,
                "market_median": predicted_salary,
                "market_range": predicted_range,
                "within_range": min_market <= user_budget <= max_market
            },
            "recommendation": recommendation,
            "confidence": confidence
        }

# Global instances
salary_collector = SalaryDataCollector()
salary_predictor = SalaryPredictor()
competitiveness_analyzer = CompetitivenessAnalyzer()

def initialize_salary_predictor():
    """Initialize the salary predictor by training or loading the model"""
    try:
        # Try to load existing model
        salary_predictor.load_model(salary_predictor.model_path)
        logger.info("Salary predictor model loaded successfully")
        return True
    except:
        logger.info("No existing model found. Collecting data and training new model...")
        try:
            # Collect data and train model
            training_data, quality_report = salary_collector.collect_strategic_data()
            if len(training_data) >= 10:
                evaluation = salary_predictor.train_model(training_data)
                logger.info(f"Salary predictor trained successfully: R² = {evaluation['r2']:.3f}")
                return True
            else:
                logger.warning("Insufficient training data collected")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize salary predictor: {e}")
            return False

# Initialize the predictor when the module is imported
initialize_salary_predictor()