import cv2
import base64

def encode_image(img, quality = 30):
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    encoded = base64.b64encode(buffer)
    return encoded