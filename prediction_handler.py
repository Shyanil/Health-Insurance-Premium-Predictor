from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model types enum
class ModelType(str, Enum):
    XGBOOST = "xgboost"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear"
    POLYNOMIAL_REGRESSION = "polynomial"

# Define input model
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    model_type: ModelType

# Load models
def load_models():
    models = {}
    try:
        # Load XGBoost model
        xgb_model = xgb.Booster()
        xgb_model.load_model("best_xgboost_model.json")
        models['xgboost'] = xgb_model
        
        # Load Decision Tree model
        dt_model = joblib.load("DecisionTree_model.pkl")
        models['decision_tree'] = dt_model
        
        # Load Random Forest model
        rf_model = joblib.load("RandomForest_model.pkl")
        models['random_forest'] = rf_model
        
        # Load Linear Regression model
        lr_model = joblib.load("LinearRegression_model.pkl")
        models['linear'] = lr_model
        
        # Load Polynomial Regression model and its preprocessors
        poly_model = joblib.load("insurance_Model.pkl")
        models['polynomial'] = {
            'model': poly_model,
            'poly_transformer': joblib.load("Final_Poly_Transformer.pkl"),
            'scaler': joblib.load("Final_Scaler.pkl")
        }
        
        logger.info("Models loaded successfully!")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

# Initialize encoders and scalers
def initialize_preprocessors():
    categorical_features = ['sex', 'smoker', 'region']
    df = pd.DataFrame([
        {'sex': 'male', 'smoker': 'yes', 'region': 'northeast'},
        {'sex': 'female', 'smoker': 'no', 'region': 'southwest'},
        {'sex': 'male', 'smoker': 'no', 'region': 'southeast'},
        {'sex': 'female', 'smoker': 'yes', 'region': 'northwest'}
    ])
    
    # Initialize and fit OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(df[categorical_features])
    
    # Initialize StandardScaler for linear regression
    scaler = StandardScaler()
    
    return encoder, scaler, categorical_features

# Global variables
MODELS = load_models()
encoder, scaler, categorical_features = initialize_preprocessors()

# Prediction function
def make_prediction(model, X_user, model_type):
    try:
        if model_type == ModelType.POLYNOMIAL_REGRESSION:
            # For Polynomial Regression, apply polynomial transformation and scaling
            poly_transformer = model['poly_transformer']
            scaler = model['scaler']
            model = model['model']
            
            X_user_poly = poly_transformer.transform(X_user)
            X_user_scaled = scaler.transform(X_user_poly)
            return model.predict(X_user_scaled)
            
        elif model_type == ModelType.LINEAR_REGRESSION:
            # For Linear Regression, apply StandardScaler
            X_user_scaled = scaler.fit_transform(X_user)
            return model.predict(X_user_scaled)
            
        elif model_type == ModelType.XGBOOST:
            dtest = xgb.DMatrix(X_user)
            return model.predict(dtest)
            
        else:  # Decision Tree or Random Forest
            return model.predict(X_user)
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Preprocessing function
def preprocess_input(user_df):
    try:
        # Apply one-hot encoding
        encoded_columns = encoder.transform(user_df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_columns,
            columns=encoder.get_feature_names_out(categorical_features)
        )
        
        # Combine numerical and encoded features
        X_user = np.concatenate([
            user_df.drop(categorical_features, axis=1).values,
            encoded_df.values
        ], axis=1)
        
        return X_user
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

# Prediction endpoint
@app.post("/predict")
async def predict_insurance(input_data: InsuranceInput):
    try:
        if not MODELS:
            raise HTTPException(status_code=500, detail="Models not loaded properly")

        # Convert input to DataFrame
        user_input = {
            'age': input_data.age,
            'sex': input_data.sex.lower(),
            'bmi': input_data.bmi,
            'children': input_data.children,
            'smoker': input_data.smoker.lower(),
            'region': input_data.region.lower()
        }
        user_df = pd.DataFrame([user_input])
        logger.debug(f"User input DataFrame:\n{user_df}")

        # Preprocess input
        X_user = preprocess_input(user_df)
        logger.debug(f"Preprocessed input:\n{X_user}")

        # Select model based on input
        model = MODELS.get(input_data.model_type)
        if not model:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {input_data.model_type}"
            )

        # Make prediction
        log_predicted_charge = make_prediction(model, X_user, input_data.model_type)
        predicted_charge = np.expm1(log_predicted_charge)
        logger.debug(f"Predicted charge: {predicted_charge}")

        # Ensure the prediction is valid
        if np.isnan(predicted_charge).any() or np.isinf(predicted_charge).any():
            raise HTTPException(
                status_code=500,
                detail="Prediction resulted in an invalid value (NaN or infinity)."
            )

        return {
            "model_type": input_data.model_type,
            "prediction": round(float(predicted_charge[0]), 2)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()) if MODELS else []
    }

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)