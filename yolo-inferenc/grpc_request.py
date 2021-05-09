import os

import cv2
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from utility import non_max_suppression, detections_boxes, \
    convert_to_original_size

import utils as utils


# Create stub for request
server = '0.0.0.0:8500'
host, port = server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()

# Model name you put on the dokcer serving image
request.model_spec.name = 'serving_model'


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

        return image_paded, gt_boxes


def get_yolov4(image):
    # inintial config
    ANCHORS = utils.get_anchors('./data/yolov4_anchors.txt', False)
    NUM_CLASS = len(utils.read_class_names('data/coco.names'))
    XYSCALE = [1.2, 1.1, 1.05]
    STRIDES = np.array([8, 16, 32])
    input_size = 608

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # send grpc request
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(
        image_data,
        dtype=types_pb2.DT_FLOAT)
    )

    result_final = []
    result_future = stub.Predict(request, 10)
    result_1 = result_future.outputs['tf_op_layer_concat_10']
    result_final.append(np.reshape(np.array(result_1.ListFields()[2][1]), \
                                   (1, 76, 76, 3, 6)
                                   ))
    result_2 = result_future.outputs['tf_op_layer_concat_11']
    result_final.append(np.reshape(np.array(result_2.ListFields()[2][1]), \
                                   (1, 38, 38, 3, 6)
                                   ))
    result_3 = result_future.outputs['tf_op_layer_concat_12']
    result_final.append(np.reshape(np.array(result_3.ListFields()[2][1]), \
                                   (1, 19, 19, 3, 6)
                                   ))

    pred_bbox = utils.postprocess_bbbox(result_final, ANCHORS, STRIDES, XYSCALE)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.45)
    bboxes = utils.nms(bboxes, 0.213, method='nms')

    #image = utils.draw_bbox(original_image, bboxes)
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #cv2.imwrite('result.jpg', image)

    return bboxes


if __name__ == '__main__':
    get_yolov4()
