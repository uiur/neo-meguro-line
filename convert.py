import cv2
import numpy as np
import argparse
import base64
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials)


def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of dicts with information about the faces in the picture.
    """
    image_content = face_file.read()
    batch_request = [{
        'image': {
            'content': base64.b64encode(image_content).decode('utf-8')
            },
        'features': [{
            'type': 'FACE_DETECTION',
            'maxResults': max_results,
            }]
        }]

    service = get_vision_service()
    request = service.images().annotate(body={
        'requests': batch_request,
        })
    response = request.execute()

    return response['responses'][0]['faceAnnotations']


def draw_black_line(image, positions):
    type_to_position = {}
    for position in positions:
        p = position['position']
        for k, v in p.items():
            p[k] = int(v)

        type_to_position[position['type']] = p

    left_top = type_to_position['LEFT_EYE_TOP_BOUNDARY']
    left_top['x'] = type_to_position['LEFT_OF_LEFT_EYEBROW']['x']

    left_bottom = type_to_position['LEFT_EYE_BOTTOM_BOUNDARY']
    left_bottom['x'] = left_top['x']

    right_bottom = type_to_position['RIGHT_EYE_BOTTOM_BOUNDARY']
    right_bottom['x'] = type_to_position['RIGHT_OF_RIGHT_EYEBROW']['x']

    right_top = type_to_position['RIGHT_EYE_TOP_BOUNDARY']
    right_top['x'] = right_bottom['x']

    cv2.fillPoly(image, [np.array([
        [left_top['x'], left_top['y']],
        [left_bottom['x'], left_bottom['y']],
        [right_bottom['x'], right_bottom['y']],
        [right_top['x'], right_top['y']],
    ])], color=(0, 0, 0), lineType=cv2.CV_AA)

parser = argparse.ArgumentParser()
parser.add_argument('image', help='a path to image')
args = parser.parse_args()

image = cv2.imread(args.image)
data = detect_face(open(args.image), 15)

for annotation in data:
    draw_black_line(image, annotation['landmarks'])

flag, buf = cv2.imencode('.png', image)
print(buf.tobytes())
