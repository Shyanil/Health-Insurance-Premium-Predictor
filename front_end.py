import streamlit as st
import streamlit.components.v1 as components

# Define enhanced chatbot UI HTML with JavaScript
chatbot_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Health insurance Premium Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f5;
        text-align: center;
        margin: 0;
        padding: 20px;
      }
      .container {
        max-width: 800px;
        margin: auto;
        background: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }
      .chat-container {
        width: 100%;
        max-width: 600px;
        background: #ffffff;
        margin: auto;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e4e8;
      }
      .chat-box {
        height: 400px;
        overflow-y: auto;
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e1e4e8;
        margin-bottom: 15px;
      }
      .chat-input {
        display: flex;
        gap: 10px;
        margin-top: 15px;
      }
      .chat-input input {
        flex: 1;
        padding: 12px;
        border: 2px solid #0066cc;
        border-radius: 8px;
        font-size: 16px;
        transition: all 0.3s ease;
      }
      .chat-input input:focus {
        outline: none;
        border-color: #0052cc;
        box-shadow: 0 0 0 2px rgba(0,102,204,0.2);
      }
      .chat-input button {
        background: #0066cc;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        transition: background-color 0.3s ease;
      }
      .chat-input button:hover {
        background: #0052cc;
      }
      .message {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 80%;
        animation: fadeIn 0.5s ease;
        position: relative;
        clear: both;
      }
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .bot-message {
        background: #f0f2f5;
        float: left;
        color: #1a1a1a;
      }
      .user-message {
        background: #0066cc;
        color: white;
        float: right;
      }
      .dropdown-container {
        margin: 20px 0;
        text-align: left;
      }
      select {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        border: 2px solid #0066cc;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        margin-top: 8px;
      }
      .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        text-align: left;
      }
      .success-message {
        background: #ebf5eb;
        color: #2e7d32;
      }
      .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #0066cc;
        animation: spin 1s linear infinite;
        margin-right: 10px;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      .clearfix::after {
        content: "";
        clear: both;
        display: table;
      }
       #myBtn {
            position: fixed;
            bottom: 20px;
            right: 0; /* Completely attached to the right */
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px 30px;
            border: none;
            border-radius: 30px 0 0 30px; /* Rounded on left side only */
            box-shadow: -5px 10px 20px rgba(255, 75, 43, 0.4);
            transition: all 0.3s ease-in-out;
            text-decoration: none;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 1000;
            width: 200px; /* Ensures proper alignment */
            text-align: center;
        }

        #myBtn:hover {
            background: linear-gradient(135deg, #ff4b2b, #ff416c);
            box-shadow: -5px 15px 30px rgba(255, 75, 43, 0.6);
            transform: translateX(-5px);
        }

        #myBtn:active {
            transform: scale(0.95);
        }
    </style>
  </head>
      <a href="https://huggingface.co/spaces/Shyanil/Advanced_Insurance_Data_Analysis_Dashboard" target="_blank" id="myBtn" title="Go to Dashboard">🚀 Visit Dashboard</a>

  <body>
    <div class="container">
      <h1 style="color: #0066cc; margin-bottom: 30px;">Health insurance Premium Predictor</h1>
      <div class="dropdown-container">
        <label for="model-select"><strong>Select Prediction Model:</strong></label>
        <select id="model-select" onchange="clearChat()">
        <option value="xgboost">BoostraX (Recommended - XGBoost)</option>
        <option value="random_forest">ForéSense (Random Forest)</option>
        <option value="decision_tree">DecisivTree (Decision Tree)</option>
        <option value="linear">LineaMind (Linear Model)</option>
        <option value="polynomial">PolyGenius (Polynomial Model)</option>
        </select>
      </div>
      <div class="chat-container">
        <div class="chat-box" id="chat-box"></div>
        <div class="chat-input">
          <input type="text" id="user-input" placeholder="Type your answer here..." onkeypress="handleKeyPress(event)"/>
          <button onclick="sendMessage()">Send</button>
        </div>
      </div>
    </div>
    <script>
      let currentStep = 0;
      let userInputs = {};
      const questions = [
          {
          text: "What is your name?",
          validation: (value) => {
            return value.trim().length > 0 ? null : "Please enter a valid name";
          },
          key: "name"
        },
        {
          text: "What is your age?",
          validation: (value) => {
            const age = parseInt(value);
            return age >= 18 && age <= 100 ? null : "Please enter a valid age between 18 and 100";
          }
        },
        {
          text: "What is your gender (male/female)?",
          validation: (value) => {
            const gender = value.toLowerCase();
            return ['male', 'female'].includes(gender) ? null : "Please enter either 'male' or 'female'";
          }
        },
        {
          text: "What is your BMI?",
          validation: (value) => {
            const bmi = parseFloat(value);
            return bmi >= 10 && bmi <= 50 ? null : "Please enter a valid BMI between 10 and 50";
          }
        },
        {
          text: "How many children do you have?",
          validation: (value) => {
            const children = parseInt(value);
            return children >= 0 && children <= 10 ? null : "Please enter a valid number between 0 and 10";
          }
        },
        {
          text: "Are you a smoker (yes/no)?",
          validation: (value) => {
            const answer = value.toLowerCase();
            return ['yes', 'no'].includes(answer) ? null : "Please enter either 'yes' or 'no'";
          }
        },
        {
          text: "What is your region (northeast/northwest/southeast/southwest)?",
          validation: (value) => {
            const region = value.toLowerCase();
            return ['northeast', 'northwest', 'southeast', 'southwest'].includes(region) 
              ? null 
              : "Please enter one of: northeast, northwest, southeast, southwest";
          }
        }
      ];
      // Function to handle key press (e.g., Enter key)
      function handleKeyPress(event) {
        if (event.key === "Enter") {
          sendMessage();
        }
      }
      // Function to get prediction from the back-end
      async function getPrediction(userData) {
        console.log(document.getElementById("model-select").value) ;
        try {
          // Show loading message
          const loadingMessage = `<div class='message bot-message clearfix'><div class='loading'></div>Calculating prediction...</div>`;
          const chatBox = document.getElementById("chat-box");
          chatBox.innerHTML += loadingMessage;
          chatBox.scrollTop = chatBox.scrollHeight;
          
          // Send data to the back-end
          const response = await fetch('https://shyanil-insurance-predictor-api.hf.space/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              age: parseInt(userData['What is your age?']),
              sex: userData['What is your gender (male/female)?'].toLowerCase(),
              bmi: parseFloat(userData['What is your BMI?']),
              children: parseInt(userData['How many children do you have?']),
              smoker: userData['Are you a smoker (yes/no)?'].toLowerCase(),
              region: userData['What is your region (northeast/northwest/southeast/southwest)?'].toLowerCase(),
              model_type: document.getElementById("model-select").value
            })
          });
          // Handle errors
          if (!response.ok) {
            throw new Error('Prediction service error');
          }
          // Parse and return the prediction
          const data = await response.json();
          return data.prediction;
        } catch (error) {
          console.error('Error:', error);
          return 'Error: Unable to get prediction. Please try again.';
        }
      }
      // Function to send user input and handle responses
      function sendMessage() {
        const userInput = document.getElementById("user-input");
        const chatBox = document.getElementById("chat-box");
        const input = userInput.value.trim();
        // Ignore empty input
        if (input === "") return;
        // Add user message to the chat box
        const userMessage = `<div class='message user-message clearfix'><strong>You:</strong> ${input}</div>`;
        chatBox.innerHTML += userMessage;
        // Validate input
        const validationError = questions[currentStep].validation(input);
        if (validationError) {
          const errorMessage = `<div class='message bot-message error-message clearfix'><strong>Error:</strong> ${validationError}</div>`;
          chatBox.innerHTML += errorMessage;
          chatBox.scrollTop = chatBox.scrollHeight;
          return;
        }
        // Store user input
        userInputs[questions[currentStep].text] = input;
        userInput.value = "";
        // Move to the next question or get prediction
        if (currentStep < questions.length - 1) {
          currentStep++;
          setTimeout(() => {
            const botMessage = `<div class='message bot-message clearfix'><strong>Bot:</strong> ${questions[currentStep].text}</div>`;
            chatBox.innerHTML += botMessage;
            chatBox.scrollTop = chatBox.scrollHeight;
          }, 500);
        } else {
          setTimeout(async () => {
            const selectedModel = document.getElementById("model-select").value;
            const prediction = await getPrediction(userInputs);
            const botMessage = `<div class='message bot-message success-message clearfix'>
              <strong>Prediction Complete!</strong><br>
              Based on your inputs, using the ${selectedModel.toUpperCase()} model:<br>
              Estimated Insurance Premium: <strong>$${typeof prediction === 'number' ? prediction.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : 'Error'}</strong>
              <br><br>
              Would you like to try another prediction? Click the "Select Model" dropdown to start over.
            </div>`;
            chatBox.innerHTML += botMessage;
            chatBox.scrollTop = chatBox.scrollHeight;
          }, 500);
        }
      }
      function clearChat() {
        const chatBox = document.getElementById("chat-box");
        const userInput = document.getElementById("user-input");
        chatBox.innerHTML = "";
        userInput.value = "";
        currentStep = 0;
        userInputs = {};
        
      setTimeout(() => {
          const userRequirements = `
              <strong>⚠️ Read the information carefully, otherwise the model will not work!</strong><br><br>
              <strong>🔹 User Input Requirements for Insurance Cost Prediction:</strong><br><br>
              <strong>1️⃣ Age:</strong> The age of the primary beneficiary. <br>
              ➤ <strong>Definition:</strong> Represents the individual's age.<br>
              ➤ <strong>Input Format:</strong> Positive integer (e.g., <strong>25</strong>, <strong>40</strong>, <strong>60</strong>).<br><br>
              <strong>2️⃣ Sex:</strong> Gender of the insurance contractor. <br>
              ➤ <strong>Definition:</strong> Indicates whether the beneficiary is male or female.<br>
              ➤ <strong>Input Format:</strong> Choose one of the following:<br>
              🔹 <strong>male</strong><br>
              🔹 <strong>female</strong><br><br>
              <strong>3️⃣ BMI (Body Mass Index):</strong> A measure indicating body weight relative to height. <br>
              ➤ <strong>Definition:</strong> Body weight divided by height squared (kg/m²).<br>
              ➤ <strong>Ideal Range:</strong> <strong>18.5 to 24.9</strong>.<br>
              ➤ <strong>Input Format:</strong> Decimal number (e.g., <strong>22.5</strong>, <strong>27.8</strong>).<br><br>
              <strong>4️⃣ Children:</strong> The number of dependents covered under the health insurance. <br>
              ➤ <strong>Definition:</strong> Number of children/dependents under the policy.<br>
              ➤ <strong>Input Format:</strong> Non-negative integer (e.g., <strong>0</strong>, <strong>1</strong>, <strong>2</strong>, <strong>3</strong>).<br><br>
              <strong>5️⃣ Smoker:</strong> Indicates whether the beneficiary is a smoker. <br>
              ➤ <strong>Definition:</strong> Smoking status of the individual.<br>
              ➤ <strong>Input Format:</strong> Choose one of the following:<br>
              🔹 <strong>yes (Smoker)</strong><br>
              🔹 <strong>no (Non-smoker)</strong><br><br>
              <strong>6️⃣ Region:</strong> The geographical residential area of the beneficiary within the United States. <br>
              ➤ <strong>Definition:</strong> The region where the person lives.<br>
              ➤ <strong>Input Format:</strong> Choose one of the following:<br>
              🔹 <strong>northeast</strong><br>
              🔹 <strong>southeast</strong><br>
              🔹 <strong>southwest</strong><br>
              🔹 <strong>northwest</strong><br><br>
              <strong>7️⃣ Charges:</strong> The estimated individual medical costs billed by health insurance. <br>
              ➤ <strong>Definition:</strong> This is the predicted output based on the provided inputs.<br>
              ➤ <strong>Users do not need to input this value.</strong><br><br>
              <em>By providing accurate and complete information, users will receive reliable predictions of medical insurance costs.</em>
              <br><br>
              <strong>What is your name?</strong><br><br>
          `;
          const welcomeMessage = `
              <div class='message bot-message clearfix'>
                  <strong>Welcome to the Health Insurance Premium Predictor!</strong><br>
                  I'll help you estimate your insurance premium. Please answer a few questions about yourself.<br><br>
                  ${userRequirements}
              </div>`;
          chatBox.innerHTML += welcomeMessage;
          chatBox.scrollTop = chatBox.scrollHeight;
      }, 300);
      }
      // Initial welcome message
      clearChat();
    </script>
  </body>
</html>
"""

# Streamlit configuration and rendering
st.set_page_config(
    page_title="Health insurance Premium Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Streamlit
st.markdown("""
    <style>
    body {
        background-color: white !important;
    }
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        background-color: white;
    }
    <style>
    body {
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Render the chatbot
components.html(chatbot_html, height=900, scrolling=True)
