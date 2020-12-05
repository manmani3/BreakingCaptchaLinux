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
TCP_PORT = 1234

# open tcp socket and wait
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', TCP_PORT))

while True :
    s.listen(True)
    conn, addr = s.accept()

    print("accpet") 
    print(addr)

    # recive 16bit
    category = int(recvall(conn, 16))
    print('received category:', category)

    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = numpy.frombuffer(stringData, dtype='uint8')

    decimg=cv2.imdecode(data, 1)
    cv2.imwrite('222.png', decimg)

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.WEIGHTS = "./model_final_1204.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15

    predictor = DefaultPredictor(cfg)
    outputs = predictor(decimg)

    print('-----------------------------------------------------')
    print('total instances:', outputs["instances"])
    print('-----------------------------------------------------')
    print('### stop running server ###')

    # get inference information
    instances = outputs['instances']
    """
    titleInstance = instances[instances.pred_classes >= 8]
    if (len(titleInstance) == 0):
        raise Exception('There is no title text')

    titleClass = titleInstance.pred_classes.numpy()[0]
    titleBox = titleInstance.pred_boxes.tensor.numpu()
    titleLeftBottomOffest = (titleBox[0][0], titleBox[0][3])
    """
    targetInstances = instances[instances.pred_classes == category]
    targetInstancesMasks = targetInstances.pred_masks.numpy()

    result = {}
    # tuple format. ex) (23.145, 35.222)
    # result['titleLeftBottomOffset'] = titleLeftBottomOffset
    # 3D array. each 2D array is mask for each one instance
    result['targetInstancesMasks'] = targetInstancesMasks
    print('target Instance:', len(targetInstancesMasks))
    print(result)
    # return inference result to client

    import pickle
    resultData = pickle.dumps(result)

    conn.send(str(len(resultData)).ljust(16).encode('utf-8'))
    conn.send(resultData)
    conn.close()
s.close()
