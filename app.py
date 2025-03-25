from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import pandas as pd
import io
import joblib

app = Flask(__name__)

# ✅ Global Load for Rainfall Dataset and Model
rainfall_df = pd.read_csv('data/Rainfall.csv')
rainfall_df.fillna(rainfall_df.mean(numeric_only=True), inplace=True)
rainfall_model = joblib.load('models/rainfall_forecast_model.pkl')
states = rainfall_df['state'].unique().tolist()

@app.route('/')
def index():
    return render_template('index.html', states=states)

@app.route('/predictions.json')
def predictions():
    predictions_path = os.path.join(app.root_path, 'static', 'predictions.json')
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/predict_rainfall", methods=["POST"])
def predict_rainfall():
    state = request.form.get("state")
    state_data = rainfall_df[rainfall_df['state'] == state].groupby('year').mean().reset_index()

    if state_data.empty:
        return jsonify({"error": "No data for selected state."})

    forecast_input = state_data[['year', 'JF', 'MAM', 'JJAS', 'OND']]
    try:
        predicted = rainfall_model.predict(forecast_input)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"})

    state_data['Predicted_Rainfall'] = predicted
    return jsonify({
        "years": state_data['year'].tolist(),
        "predictions": state_data['Predicted_Rainfall'].round(2).tolist()
    })

@app.route('/rainfall_forecast')
def rainfall_forecast():
    year = int(request.args.get('year', 2026))  # ✅ Dynamic Year Support
    seasonal_avg = rainfall_df.groupby('state')[['JF', 'MAM', 'JJAS', 'OND']].mean().reset_index()
    future_data = pd.DataFrame({'state': states, 'year': year})
    future_data = future_data.merge(seasonal_avg, on='state', how='left')

    X_future = future_data[['year', 'JF', 'MAM', 'JJAS', 'OND']]
    try:
        predictions = rainfall_model.predict(X_future)
    except Exception as e:
        return jsonify({"error": f"Forecast prediction failed: {str(e)}"})

    future_data['annual_rainfall_predicted'] = predictions
    future_data['Drought_Status'] = future_data['annual_rainfall_predicted'].apply(
        lambda x: '🚨 Drought' if x < 800 else '✅ No Drought'
    )

    data = future_data[['state', 'annual_rainfall_predicted', 'Drought_Status']].to_dict(orient='records')
    return jsonify(data)

@app.route('/download_rainfall', methods=['GET'])
def download_rainfall():
    year = int(request.args.get('year', 2026))  # ✅ Optional year for download
    seasonal_avg = rainfall_df.groupby('state')[['JF', 'MAM', 'JJAS', 'OND']].mean().reset_index()
    future_data = pd.DataFrame({'state': states, 'year': year})
    future_data = future_data.merge(seasonal_avg, on='state', how='left')

    X_future = future_data[['year', 'JF', 'MAM', 'JJAS', 'OND']]
    try:
        predictions = rainfall_model.predict(X_future)
    except Exception as e:
        return jsonify({"error": f"Download prediction failed: {str(e)}"})

    future_data['annual_rainfall_predicted'] = predictions
    future_data['Drought_Status'] = future_data['annual_rainfall_predicted'].apply(
        lambda x: '🚨 Drought' if x < 800 else '✅ No Drought'
    )

    output = io.StringIO()
    future_data[['state', 'annual_rainfall_predicted', 'Drought_Status']].to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        download_name=f'Rainfall_Prediction_{year}.csv',
        as_attachment=True
    )

@app.route('/state_month_rainfall')
def state_month_rainfall():
    state = request.args.get('state')
    if not state:
        return jsonify({"error": "State parameter is required."})

    state_data = rainfall_df[rainfall_df['state'] == state].sort_values(by='year', ascending=False)
    if state_data.empty:
        return jsonify({"error": "No data found for this state."})

    latest = state_data.iloc[0]
    month_data = {month: latest.get(month, 0) for month in
                  ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']}
    return jsonify(month_data)

if __name__ == '__main__':
    app.run(debug=True)
