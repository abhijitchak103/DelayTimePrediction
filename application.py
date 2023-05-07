from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)


app = application


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            Delivery_person_Age= float(request.form.get('Delivery_person_Age')), 
            Delivery_person_Ratings= float(request.form.get('Delivery_person_Ratings')), 
            Vehicle_condition= float(request.form.get('Vehicle_condition')),
            multiple_deliveries= float(request.form.get('multiple_deliveries')), 
            Festival= request.form.get('Festival'), 
            Delivery_distance= float(request.form.get('Delivery_distance')), 
            Time_to_pick= float(request.form.get('Time_to_pick')), 
            Weather_conditions= request.form.get('Weather_conditions'),
            Road_traffic_density= request.form.get('Road_traffic_density'),
            Type_of_order= request.form.get('Type_of_order'), 
            Type_of_vehicle= request.form.get('Type_of_vehicle'),
            City= request.form.get('City'), 
            Time_of_Day_Ordered= request.form.get('Time_of_Day_Ordered'),
            Month= request.form.get('Month')
        )

        final_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict_data(final_df)

        result = round(pred[0], 2)

        return render_template('results.html', final_result=f"{result} minutes")
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = False)