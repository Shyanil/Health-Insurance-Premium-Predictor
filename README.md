# **Health Insurance Premium Predictor**

<img src="https://github.com/Shyanil/Health-Insurance-Premium-Predictor/blob/main/Flowchart%20-%20Frame%201.jpg" alt="Health Insurance Premium Predictor" width="100%">

## **Live Demo & API**

- 🚀 **Live Testing Link:** [Health Insurance Premium Predictor](https://huggingface.co/spaces/Shyanil/Health_insurance_Premium_Predictor)
- 🖥 **Backend API:** [Insurance Predictor API](https://huggingface.co/spaces/Shyanil/insurance-predictor-api?logs=container)
- 🎮 **Play with Data:** [Interactive Data Exploration](https://huggingface.co/spaces/Shyanil/Advanced_Insurance_Data_Analysis_Dashboard)

This project integrates the entire machine learning pipeline for predicting health insurance premiums using multiple models, extensive data analysis, and interactive UI.

---

## **📌 Project Journey & Contribution Guidelines**

If you wish to contribute, follow this step-by-step guide.

---

## **🔍 1. Data Analysis & Preprocessing**

- Conducted extensive **exploratory data analysis (EDA)** using scatter plots, box plots, 3D visualizations, KDE plots, heatmaps, and more.
- Interactive data visualizations available **[here](https://randomlink.com)**.
- **Feature Engineering:**
  - One-hot encoding for categorical variables (**sex, smoker, region**).
  - **Log transformation** applied to target variable `charges` using `np.log1p()`.
- **Dataset Cleaning:** Handled missing values, outliers, and performed feature scaling.
- **Libraries Used:** pandas, numpy, seaborn, matplotlib, scikit-learn, optuna, streamlit, gradio, huggingface
- **Detailed analysis available in** `insurance_data_analysis.ipynb`.

---

## **🤖 2. Model Selection & Performance Evaluation**

### **A. Preprocessing & Model Testing**

Tested multiple models: **XGBoost, Decision Tree, Random Forest, Linear Regression, Polynomial Regression**.

### **B. Linear Regression**

Trained after preprocessing, yielding:

- **Mean Squared Error (MSE):** 0.1746
- **Scaled MSE (MSE/2):** 0.0873
- **Mean Absolute Error (MAE):** 0.2685

### **C. Polynomial Regression (Degree = 2)**

After testing multiple degrees, best results with **degree = 2**:

- **MSE:** 0.1207
- **Scaled MSE:** 0.0604
- **MAE:** 0.2049

**Saved in** `insurance_parametric_regression.ipynb`.

### **D. Decision Tree & Random Forest**

**Hyperparameter tuning applied using Optuna**.

#### **Decision Tree Results:**

- **MSE:** 0.1464
- **Scaled MSE:** 0.0732
- **MAE:** 0.2188

#### **Random Forest Results:**

- **MSE:** 0.1302
- **Scaled MSE:** 0.0651
- **MAE:** 0.2073

(See `insurance_Decision_tree.ipynb` for details.)

### **E. Final Model: XGBoost (Optimized with Optuna)**

- **MSE:** 0.1278
- **Scaled MSE:** 0.0639
- **MAE:** 0.2022

**Final model saved in** `XGBoost_insurance_model.ipynb`.

---

## **⚙️ 3. MLOps & Deployment**

### **A. Backend API (FastAPI & Gradio)**

- Handles **incoming user input**, loads trained models, and returns predictions.
- Uses **FastAPI** and **Gradio** for deployment.
- **Backend triggers on input, loads models via joblib, and returns results.**
- Deployed on **Hugging Face Spaces**.

#### **Backend Features:**

✅ **FastAPI Setup:** Handles API requests and CORS.
✅ **Model Loading:** Supports XGBoost, Decision Tree, Random Forest, Linear Regression, and Polynomial Regression.
✅ **Data Preprocessing:** One-hot encoding of categorical variables.
✅ **Prediction Handling:** Accepts input, preprocesses, selects the model, and returns predictions.
✅ **Endpoints:**

- `/` - Root endpoint (Welcome message).
- `/predict` - Accepts data and returns predictions.
- `/health` - Health check endpoint.
  ✅ **Logging & Error Handling:** Ensures smooth debugging.
  ✅ **Cross-Origin Compatibility:** Allows frontend to communicate via CORS.
  ✅ **Deployment:** Hosted on Hugging Face Spaces.

**Backend file:** `prediction_handler.py`

---

## **💻 4. Frontend (Streamlit UI & Gradio)**

Developed an **interactive chatbot-style UI** for **Health Insurance Premium Prediction**.

### **A. Technologies Used**

- **Streamlit** - Web framework.
- **Gradio** - Interactive UI framework.
- **HTML, CSS, JavaScript** - UI enhancements.
- **Fetch API** - Calls backend FastAPI.
- **ML Models:** Supports XGBoost, Decision Tree, Random Forest, Linear Regression, and Polynomial Regression.

### **B. Features**

🎨 **Modern UI:** Clean and intuitive design with custom styling.
📊 **Dropdown for Model Selection:** Users dynamically select models.
✅ **Validation Checks:**

- Ensures valid age (18-100), BMI range, and categorical values.
  ⏳ **Loading Animation & User Interaction:** Enhances user experience.

**Frontend file:** `front_end.py`

---

## **🔮 5. Future Improvements**

🔹 **Hyperparameter tuning with Bayesian Optimization**
🔹 **Integration with cloud platforms (AWS/GCP) for scalable deployment**
🔹 **Mobile-friendly UI enhancements**
🔹 **Adding more ML models and explainable AI (SHAP, LIME)**

📢 **Want to contribute?** Follow the structure in the respective `.ipynb` files and reach out via the repository issues section!

---

**📝 Final Note:**  
For a **better understanding** of the project workflow, **open and explore** each file. I have added **detailed comments** in every script and notebook to guide you through each step, from data preprocessing to model deployment. These comments will help you follow the logic behind every decision and modification. 🚀

Let me know if you'd like any further refinements!

## **📜 License & Acknowledgments**

This project is open-source and licensed under **MIT License**.

Special thanks to:

- **Hugging Face Spaces** for hosting.
- **Optuna** for hyperparameter tuning.
- **Streamlit & FastAPI** for interactive UI and API.
- **Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn** for data processing and visualization.
- **Community Contributors** for improvements.

💡 **Stay tuned for updates! 🚀**

