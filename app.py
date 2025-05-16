from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv('healthcare_data.csv')
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    recency = float(request.form['recency'])
    frequency = float(request.form['frequency'])
    monetary = float(request.form['monetary'])
    time = float(request.form['time'])

    input_data = pd.DataFrame([{
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Time': time
    }])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 0:
        message = "No immediate action needed. Maintain your current health routine."
    else:
        message = "Recommendation: Please consult a doctor or follow up for medical attention."

    return render_template('index.html', prediction_text=message)

if __name__ == "__main__":
    app.run(debug=True)
