from main import manual_testing
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json.get('user_input')  # Get data sent from JavaScript
    response_text = manual_testing(data)  # Process the input
    return jsonify(response=response_text)  # Send response back to frontend

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Analyze')
def Analyze():
    return render_template('Analyze.html')


if __name__ == '__main__':
    app.run(debug=True)
