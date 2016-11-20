import io
import logging
from flask import Flask, request, send_file, render_template
import cv2
import convert
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        # limit is 4MB for Cloud Vision Api
        f = request.files['image']

        image_bytearray = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytearray, 1)
        data = convert.detect_face(image, 15)
        for annotation in data:
            convert.draw_black_line(image, annotation['landmarks'])

        return send_file(
            io.BytesIO(convert.image_to_bytes(image)), mimetype='image/png'
        )

    else:
        return render_template('index.html')


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
    app.run(host='0.0.0.0', port=8080, debug=True)
