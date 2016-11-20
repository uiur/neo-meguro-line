import io
import logging
from flask import Flask, request, send_file, render_template
import cv2
import convert
import numpy as np
from PIL import Image, JpegImagePlugin

app = Flask(__name__)


def rotate_if_needed(bytes):
    convert_image = {
        1: lambda img: img,
        2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        3: lambda img: img.transpose(Image.ROTATE_180),
        4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
        6: lambda img: img.transpose(Image.ROTATE_270),
        7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
        8: lambda img: img.transpose(Image.ROTATE_90),
    }

    img = Image.open(io.BytesIO(bytes))

    # cloud vision api doesn't accept large image
    img.thumbnail((1600, 1600), Image.ANTIALIAS)
    new_img = img

    if img.format == "JPEG":
        exif = img._getexif()
        if exif:
            orientation = exif.get(0x112, 1)

            new_img = convert_image[orientation](img)

    return cv2.cvtColor(np.array(new_img), cv2.COLOR_BGR2RGB)


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        # limit is 4MB for Cloud Vision Api
        f = request.files['image']

        image = rotate_if_needed(f.read())
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
