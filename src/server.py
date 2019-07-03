from flask import Flask, jsonify, request, render_template

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


@app.route('/predict', methods=['GET', 'POST'])
def predictions():

    if request.method == 'GET':
        print('Making predictions...')
        predictions = {
            'probabilities': [0.0, 0.1, 0.05, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0, 0.5],
            'prediction': 9
            }
        return jsonify(predictions)
    else:
        print('Incoming bitmap...')
        bitmap = request.get_json()['bitmap']
        print('bitmap len', len(bitmap))
        print(bitmap)

        return 'OK', 200
