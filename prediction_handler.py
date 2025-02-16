from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import logging
import uvicorn
from fastapi.responses import JSONResponse

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
        
        # Load other models
        models['decision_tree'] = joblib.load("DecisionTree_model.pkl")
        models['random_forest'] = joblib.load("RandomForest_model.pkl")
        models['linear'] = joblib.load("LinearRegression_model.pkl")
        
        # Load Polynomial Regression model and preprocessors
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

# Initialize preprocessors
def initialize_preprocessors():
    categorical_features = ['sex', 'smoker', 'region']
    df = pd.DataFrame([
        {'sex': 'male', 'smoker': 'yes', 'region': 'northeast'},
        {'sex': 'female', 'smoker': 'no', 'region': 'southwest'},
        {'sex': 'male', 'smoker': 'no', 'region': 'southeast'},
        {'sex': 'female', 'smoker': 'yes', 'region': 'northwest'}
    ])
    
    # ✅ Fixed `sparse_output=False` for latest scikit-learn
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(df[categorical_features])

    return encoder, categorical_features

# Global variables
MODELS = load_models()
encoder, categorical_features = initialize_preprocessors()

# Prediction function
def make_prediction(model, X_user, model_type):
    try:
        if model_type == ModelType.POLYNOMIAL_REGRESSION:
            poly_transformer = model['poly_transformer']
            poly_scaler = model['scaler']
            model = model['model']
            X_user_poly = poly_transformer.transform(X_user)
            X_user_scaled = poly_scaler.transform(X_user_poly)
            return model.predict(X_user_scaled)

        elif model_type == ModelType.LINEAR_REGRESSION:
            # ✅ Fit scaler dynamically on `X_user`
            scaler_lr = StandardScaler()
            scaler_lr.fit(X_user)
            X_user_scaled = scaler_lr.transform(X_user)
            return model.predict(X_user_scaled)

        elif model_type == ModelType.XGBOOST:
            dtest = xgb.DMatrix(X_user)
            return model.predict(dtest)

        else:
            return model.predict(X_user)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Preprocessing function
def preprocess_input(user_df):
    try:
        encoded_columns = encoder.transform(user_df[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_columns,
            columns=encoder.get_feature_names_out(categorical_features)
        )
        X_user = np.concatenate([
            user_df.drop(categorical_features, axis=1).values,
            encoded_df.values
        ], axis=1)
        return X_user
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Insurance Premium Prediction API"}

# Prediction endpoint
@app.post("/predict")
async def predict_insurance(input_data: InsuranceInput):
    try:
        if not MODELS:
            raise HTTPException(status_code=500, detail="Models not loaded properly")

        user_input = {
            'age': input_data.age,
            'sex': input_data.sex.lower(),
            'bmi': input_data.bmi,
            'children': input_data.children,
            'smoker': input_data.smoker.lower(),
            'region': input_data.region.lower()
        }
        user_df = pd.DataFrame([user_input])
        X_user = preprocess_input(user_df)

        model = MODELS.get(input_data.model_type)
        if not model:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {input_data.model_type}")

        log_predicted_charge = make_prediction(model, X_user, input_data.model_type)
        predicted_charge = np.expm1(log_predicted_charge)

        response_data = {
            "model_type": input_data.model_type,
            "prediction": round(float(predicted_charge[0]), 2)
        }
        
        # Set CORS headers in response
        response = JSONResponse(content=response_data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": list(MODELS.keys()) if MODELS else []}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)