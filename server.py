#!/usr/bin/python
import socket
import cv2
import numpy


# socket recive buffer
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


TCP_IP = '127.0.0.1'
TCP_PORT = 9999

# open tcp socket and wait
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', TCP_PORT))
s.listen(True)

print('9999port waiting')

conn, addr = s.accept()

# recive 16bit
length = recvall(conn, 16)
stringData = recvall(conn, int(length))
data = numpy.frombuffer(stringData, dtype='uint8')
s.close()
decimg=cv2.imdecode(data, 1)
cv2.imwrite('test.png', decimg)

# Image Test code
"""
cv2.imshow('SERVER', decimg)
cv2.waitKey(0)
cv2.destroyAllWindows() 
"""

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.WEIGHTS = "./model_final_1201.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

predictor = DefaultPredictor(cfg)
outputs = predictor(decimg)

print(outputs["instances"])
print('### stop running server ###')
