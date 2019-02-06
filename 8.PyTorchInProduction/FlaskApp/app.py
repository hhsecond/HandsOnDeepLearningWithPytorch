import json

from flask import Flask

import controller

app = Flask('FizBuzAPI')


@app.route('/<which>')
def predict(which):
    try:
        number = int(which) + 1
        prediction = controller.run(number)
        out = json.dumps({'NextNumber': prediction})
    except ValueError:
        out = json.dumps({'NextNumber': 'WooHooo!!!'})
    return out
