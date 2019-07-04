from flask import Flask, jsonify, request, render_template

from train import Model

app = Flask(__name__)

@app.route('/hello', methods=['POST', 'GET'])
def hello():

    if request.method == 'POST':
        print('Incoming..')
        print(request.get_json())
        return 'OK', 200
    
    else:
        message = {'greetings' : 'Hello from Flask!'}
        return jsonify(message)


@app.route('/test')
def test_page():
    return render_template('index.html') #from 'templates' directory


model = Model()

@app.route('/predict', methods=['GET', 'POST'])
def predictions():
    global prediction, probabilities
    if request.method == 'GET':
        print('Making predictions...')
        predictions = {
            'probabilities': probabilities.tolist(),
            'prediction': int(prediction)
            }
        return jsonify(predictions)
    else:
        print('Incoming bitmap...')
        bitmap = request.get_json()['bitmap']
        # print('bitmap len', len(bitmap))
        # print(bitmap)

        prediction, probabilities = model.predict([bitmap])

        return 'OK', 200
