from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model_of_farmer.pkl')
label_encoder = joblib.load('crop_label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        try:
            N = int(request.form['N'])
            P = int(request.form['P'])
            K = int(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            features = [[N, P, K, temperature, humidity, ph, rainfall]]
            prediction = model.predict(features)[0]
            crop_name = label_encoder.inverse_transform([prediction])[0]

            return render_template('project.html', prediction=f"Recommended Crop: {crop_name}")

        except Exception as e:
            return render_template('project.html', prediction=f"Error: {e}")

    return render_template('project.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
