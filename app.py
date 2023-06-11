from flask import Flask, render_template, request, jsonify

from chat import get_response

import pickle
import logging

# some_file.py
#import sys
#sys.path.insert(0,'/datasets/')
from datasets.recommender.main_sample import get_recommend

app = Flask(__name__)

#app.static_folder = 'static'

# @app.get("/")
# def index_get():
#     return render_template("./home.html")

@app.get('/')
def index_get():
    return render_template("home.html")

@app.get('/home')
def home():
    return render_template("home.html")

@app.get('/cards')
def cards():
    return render_template("cards.html")


@app.get('/about')
def about():
    return render_template("about.html")

@app.route('/regression')
def regressor():
    return render_template("regression.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    recommended_item = [0,0]
    input_item = ['Null']
    app.logger.info('****Recommend**'+str(request.method))
    if request.method == 'POST':
        input_item = str(request.form['item_bought'])
        recommended_item = get_recommend(input_item)

    app.logger.info('****Recommend**'+str(recommended_item))
    if recommended_item is None:
        return render_template('recommendation.html', Bought_item='For Bought Item: {}'.format(input_item),Recommended_item='Recommended Item: {}'.format("N/A"),
        Probability='Probability Percentage: {}'.format(str("N/A")))
    else:
        app.logger.info('****Recommend**'+str(recommended_item[1]))
        probability = round(float(recommended_item[1]),2)*100
        return render_template('recommendation.html', Bought_item='For Bought Item: {}'.format(input_item),Recommended_item='Recommended Item: {}'.format(recommended_item[0]),
        Probability='Probability Percentage: {} %'.format(str(probability)))

if __name__ == "__main__":
    app.run(debug=True, port='5000')

