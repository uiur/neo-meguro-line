import cv2
import numpy as np
import argparse
import base64
import sys
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials)


def detect_face(image, max_results=4):
    image_content = image_to_bytes(image)
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


def image_to_bytes(image):
    flag, buf = cv2.imencode('.png', image)
    return buf.tobytes()


def draw_black_line(image, positions):
    PADDING_VERTICAL_RATIO = 0.5
    PADDING_HORIZONTAL_RATIO = 0.05

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

    left_height = left_bottom['y'] - left_top['y']
    left_top['y'] -= int(left_height * PADDING_VERTICAL_RATIO)
    left_bottom['y'] += int(left_height * PADDING_VERTICAL_RATIO)

    right_bottom = type_to_position['RIGHT_EYE_BOTTOM_BOUNDARY']
    right_bottom['x'] = type_to_position['RIGHT_OF_RIGHT_EYEBROW']['x']

    right_top = type_to_position['RIGHT_EYE_TOP_BOUNDARY']
    right_top['x'] = right_bottom['x']

    right_height = right_bottom['y'] - right_top['y']
    right_top['y'] -= int(right_height * PADDING_VERTICAL_RATIO)
    right_bottom['y'] += int(right_height * PADDING_VERTICAL_RATIO)

    width = right_top['x'] - left_top['x']
    left_top['x'] -= int(width * PADDING_HORIZONTAL_RATIO)
    left_bottom['x'] -= int(width * PADDING_HORIZONTAL_RATIO)

    right_top['x'] += int(width * PADDING_HORIZONTAL_RATIO)
    right_bottom['x'] += int(width * PADDING_HORIZONTAL_RATIO)

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
data = detect_face(image, 15)

for annotation in data:
    draw_black_line(image, annotation['landmarks'])

print(image_to_bytes(image))
