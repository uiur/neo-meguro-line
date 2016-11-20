import cv2
import numpy as np
import argparse
import base64
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
    first_response = response['responses'][0]
    if 'error' in first_response:
        print(first_response['error'])
        raise

    if 'faceAnnotations' not in first_response:
        return []

    return first_response['faceAnnotations']


def image_to_bytes(image):
    flag, buf = cv2.imencode('.png', image)
    return buf.tobytes()


def point_to_vector(p):
    return np.array([p['x'], p['y']])


def draw_black_line(image, positions):
    PADDING_VERTICAL_RATIO = 1.25
    PADDING_HORIZONTAL_RATIO = 0.4

    type_to_position = {}
    for position in positions:
        p = position['position']
        for k, v in p.items():
            p[k] = int(v)

        type_to_position[position['type']] = p

    left = point_to_vector(type_to_position['LEFT_EYE'])
    right = point_to_vector(type_to_position['RIGHT_EYE'])

    left_top = np.array(left)
    left_bottom = np.array(left)

    right_top = np.array(right)
    right_bottom = np.array(right)

    horizontal_direction = right - left
    normal = np.array([horizontal_direction[1], -horizontal_direction[0]], int)
    normal = normal / np.linalg.norm(normal)

    # vertical
    left_height = np.linalg.norm(point_to_vector(type_to_position['LEFT_EYE_BOTTOM_BOUNDARY']) - point_to_vector(type_to_position['LEFT_EYE_TOP_BOUNDARY']))
    right_height = np.linalg.norm(point_to_vector(type_to_position['RIGHT_EYE_BOTTOM_BOUNDARY']) - point_to_vector(type_to_position['RIGHT_EYE_TOP_BOUNDARY']))

    height = max(left_height, right_height)
    left_top += np.array(height * PADDING_VERTICAL_RATIO * normal, int)
    left_bottom -= np.array(height * PADDING_VERTICAL_RATIO * normal, int)

    right_top += np.array(height * PADDING_VERTICAL_RATIO * normal, int)
    right_bottom -= np.array(height * PADDING_VERTICAL_RATIO * normal, int)

    horizontal_pad = np.array(PADDING_HORIZONTAL_RATIO * (right - left), int)
    left_top -= horizontal_pad
    left_bottom -= horizontal_pad
    right_top += horizontal_pad
    right_bottom += horizontal_pad

    cv2.fillPoly(image, [np.array([
        left_top,
        left_bottom,
        right_bottom,
        right_top,
    ])], color=(0, 0, 0), lineType=cv2.CV_AA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='a path to image')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    data = detect_face(image, 15)

    for annotation in data:
        draw_black_line(image, annotation['landmarks'])

    print(image_to_bytes(image))
