from FlightPricePrediction.pipelines.prediction_pipeline import PredictionPipeline, FlightCustomData
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # Collect form data
        dep_time = request.form.get("dep_time")  # e.g., "10:30"
        arr_time = request.form.get("arrival_time")  # e.g., "12:30"
        
        dep_hour, dep_minute = map(int, dep_time.split(':'))
        arr_hour, arr_minute = map(int, arr_time.split(':'))

        data = {
            'Airline': request.form.get("airline"),
            'Source': request.form.get("source"),
            'Destination': request.form.get("destination"),
            'Total_Stops': request.form.get("total_stops"),
            'Dep_hour': dep_hour,
            'Dep_minute': dep_minute,
            'Arr_hour': arr_hour,
            'Arr_minute': arr_minute,
            'Duration': request.form.get("duration"),
            'Journey_day': pd.to_datetime(request.form.get("date_of_journey")).day,
            'Journey_month': pd.to_datetime(request.form.get("date_of_journey")).month
        }

        # Convert data to DataFrame
        final_data = pd.DataFrame([data])

        # Create prediction pipeline
        predict_pipeline = PredictionPipeline()

        # Make prediction
        pred = predict_pipeline.make_prediction(final_data)
        result = round(pred[0], 2)

        return render_template("result.html", final_result=result)


if __name__ == "__main__":
    app.run(debug=True)