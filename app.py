from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and dataframe
pipe = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
df = pickle.load(open(os.path.join(BASE_DIR, 'df.pkl'), 'rb'))

@app.route('/')
def index():
    companies = sorted(df['Company'].unique().tolist())
    types = sorted(df['TypeName'].unique().tolist())
    cpus = sorted(df['Cpu brand'].unique().tolist())
    gpus = sorted(df['Gpu brand'].unique().tolist())
    os_list = sorted(df['os'].unique().tolist())

    return render_template('index.html',
                           companies=companies,
                           types=types,
                           cpus=cpus,
                           gpus=gpus,
                           os_list=os_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        company = data['company']
        type_name = data['type']
        ram = int(data['ram'])
        weight = float(data['weight'])
        touchscreen = int(data['touchscreen'])
        ips = int(data['ips'])
        screen_size = float(data['screen_size'])
        resolution = data['resolution']
        cpu = data['cpu']
        hdd = int(data['hdd'])
        ssd = int(data['ssd'])
        gpu = data['gpu']
        os_name = data['os']

        # PPI calculation
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

        query = np.array([company, type_name, ram, weight, touchscreen,
                          ips, ppi, cpu, hdd, ssd, gpu, os_name]).reshape(1, 12)

        prediction = np.exp(pipe.predict(query)[0])

        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)