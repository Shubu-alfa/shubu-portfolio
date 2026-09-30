import os

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from auth_utils import admin_required, is_admin_authenticated, verify_admin_password
from chat import get_response
from datasets.recommender.main_sample import get_recommend
from db import init_db, log_exchange

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-only-change-me")

init_db()


@app.get("/")
def index_get():
    return render_template("home.html")


@app.get("/home")
def home():
    return render_template("home.html")


@app.get("/cards")
def cards():
    return render_template("cards.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.route("/regression")
def regressor():
    return render_template("regression.html")


@app.route("/linear_regression", methods=["GET", "POST"])
def linear_regressor():
    prediction_text = None
    if request.method == "POST":
        prediction_text = "Prediction demo is available after the model endpoint is wired for production."
    return render_template("linear_regression.html", prediction_text=prediction_text)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("message") or "").strip()
    client_session_id = payload.get("session_id")

    if not text:
        return jsonify({"answer": "Please type a message.", "session_id": client_session_id}), 400

    response = get_response(text)

    try:
        session_id = log_exchange(
            session_id=client_session_id,
            user_text=text,
            bot_text=response if isinstance(response, str) else str(response),
            ip=request.headers.get("X-Forwarded-For", request.remote_addr),
            user_agent=request.headers.get("User-Agent"),
        )
    except Exception as exc:
        app.logger.exception("Failed to log chat exchange: %s", exc)
        session_id = client_session_id

    return jsonify({"answer": response, "session_id": session_id})


@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    recommended_item = [0, 0]
    input_item = ["Null"]
    app.logger.info("****Recommend**" + str(request.method))
    if request.method == "POST":
        input_item = str(request.form["item_bought"])
        recommended_item = get_recommend(input_item)

    app.logger.info("****Recommend**" + str(recommended_item))
    if recommended_item is None:
        return render_template(
            "recommendation.html",
            Bought_item="For Bought Item: {}".format(input_item),
            Recommended_item="Recommended Item: {}".format("N/A"),
            Probability="Probability Percentage: {}".format(str("N/A")),
        )

    app.logger.info("****Recommend**" + str(recommended_item[1]))
    probability = round(float(recommended_item[1]), 2) * 100
    return render_template(
        "recommendation.html",
        Bought_item="For Bought Item: {}".format(input_item),
        Recommended_item="Recommended Item: {}".format(recommended_item[0]),
        Probability="Probability Percentage: {} %".format(str(probability)),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.values.get("next") or url_for("trading")
    if is_admin_authenticated():
        return redirect(next_url)

    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if verify_admin_password(password):
            session["is_admin"] = True
            session.permanent = True
            return redirect(next_url)
        error = "Incorrect password."

    return render_template("login.html", error=error, next_url=next_url)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("index_get"))


@app.get("/trading")
@admin_required
def trading():
    return render_template("trading.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
