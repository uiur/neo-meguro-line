import io
import logging
from flask import Flask, request, send_file
import cv2
import convert
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        f = request.files['image']

        image = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1)
        data = convert.detect_face(image, 15)
        for annotation in data:
            convert.draw_black_line(image, annotation['landmarks'])

        return send_file(io.BytesIO(convert.image_to_bytes(image)))

    else:
        return 'Hello World!'


@app.route('/_ah/health')
def health():
    return 'ok'


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
