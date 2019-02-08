import json

from flask import Flask
from flask import request

import controller

app = Flask('FizBuzAPI')


@app.route('/predictions/fizbuz_package', methods=['POST'])
def predict():
    which = request.get_json().get('input.1')
    if not which:
        return "InvalidData"
    try:
        number = int(which) + 1
        prediction = controller.run(number)
        out = json.dumps({'NextNumber': prediction})
    except ValueError:
        out = json.dumps({'NextNumber': 'WooHooo!!!'})
    return out
