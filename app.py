from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\aryan\model_pickle.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# In-memory history list
history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            spx = float(request.form.get('spx', 0))
            uso = float(request.form.get('uso', 0))
            slv = float(request.form.get('slv', 0))
            eurusd = float(request.form.get('eurusd', 0))
            features = np.array([[spx, uso, slv, eurusd]])
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)
            # Store in history
            history.append({
                'spx': spx,
                'uso': uso,
                'slv': slv,
                'eurusd': eurusd,
                'prediction': prediction
            })
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction, history=history[-5:])  # Show last 5

if __name__ == '__main__':
    app.run(debug=True)
