from src.mqtt.mqtt_laptop import MQTT
from src.object_detection.od import OD
from src.depth.depth import Depth
import cv2


if __name__ == "__main__":
    od = OD(model = "m")
    depth = Depth()
    mqtt = MQTT()

    while True: 
        try:
            img = mqtt.image
            dep = depth.get_depth(img)
            predict, img = od.detect(img)   
            
            cv2.imshow("res", img)
            cv2.imshow("dep", dep)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                mqtt.disconnect()
                break
            # cv2.imwrite("out.jpg", img)
        except KeyboardInterrupt:
            mqtt.disconnect()
            break