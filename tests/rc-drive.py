import threading
import SocketServer
import serial
import cv2
import numpy as np
import math

# distance data measured by ultrasonic sensor
sensor_data = " "


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, 4])
        self.model.create(layer_size)
        self.model.load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


class RCControl(object):

    def __init__(self):
        self.serial_port = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1)

    def steer(self, prediction):
        if prediction == 2:
            self.serial_port.write(chr(1))
            print("Forward")
        elif prediction == 0:
            self.serial_port.write(chr(7))
            print("Left")
        elif prediction == 1:
            self.serial_port.write(chr(6))
            print("Right")
        else:
            self.stop()

    def stop(self):
        self.serial_port.write(chr(0))


class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d

class SensorDataHandler(SocketServer.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        try:
            while self.data:
                self.data = self.request.recv(1024)
                sensor_data = round(float(self.data), 1)
                #print "{} sent:".format(self.client_address[0])
                print sensor_data
        finally:
            print "Connection closed on thread 2"


class VideoStreamHandler(SocketServer.StreamRequestHandler):

    model = NeuralNetwork()
    model.create()

    rc_car = RCControl()

    def handle(self):

        global sensor_data
        stream_bytes = ' '

        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)
                    # cv2.CV_LOAD_IMAGE_GRAYSCALE = 0
                    # cv2.CV_LOAD_IMAGE_UNCHANGED = -1

                    # lower half of the image
                    half_gray = gray[120:240, :]

                    cv2.imshow('image', image)

                    # reshape image
                    image_array = half_gray.reshape(1, 38400).astype(np.float32)
                    
                    # neural network makes prediction
                    prediction = self.model.predict(image_array)

                    # stop conditions
                    self.rc_car.steer(prediction)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.rc_car.stop()
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread('192.168.1.100', 8000))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()