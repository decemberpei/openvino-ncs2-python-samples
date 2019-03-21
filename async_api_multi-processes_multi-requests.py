#!/usr/bin/env python3

import cv2
import os
import sys
import time

import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin

from multiprocessing import Process, Queue
import multiprocessing
import threading
import queue

def async_infer_worker(exe_net, request_number, image_queue, input_blob, out_blob):
    global start_time
    
    current_request_ids = range(request_number)
    next_request_ids = range(request_number, request_number * 2)
    done = False
    last_batch = -1
    infered_images = 0
    while True:
        buffers = []
        for i in range(request_number):
            b = image_queue.get()
            if type(b) != np.ndarray:
                buffers.append(None)
                done = True
                break
            else:
                buffers.append(b)
        for _request_id in current_request_ids:
            if _request_id >=  request_number:
                if type(buffers[_request_id - request_number]) == np.ndarray:
                    exe_net.start_async(request_id=_request_id, inputs={input_blob: buffers[_request_id - request_number]})
                else:
                    #print("image at index " + str(_request_id - request_number) + " is none." )
                    last_batch = _request_id - request_number
                    break
            else:
                if type(buffers[_request_id]) == np.ndarray:
                    exe_net.start_async(request_id=_request_id, inputs={input_blob: buffers[_request_id]})
                else:
                    #print("image at index " + str(_request_id) + " is none." )
                    last_batch = _request_id
                    break
                    
        for _request_id in next_request_ids:
            if exe_net.requests[_request_id].wait(-1) == 0:
                res = exe_net.requests[_request_id].outputs[out_blob]
                infered_images = infered_images + 1
                #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
                duration = time.time() - start_time
                print("inferred images: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)

        current_request_ids, next_request_ids = next_request_ids, current_request_ids
        
        #for i in range(len(buffers)):
        #    image_queue.task_done()
            
        if done:
            break

    # 'last_batch' more inference results remain to check
    buffer_index = 0
    for _request_id in next_request_ids:
        if(buffer_index >= last_batch):
            break
        buffer_index = buffer_index + 1
        if exe_net.requests[_request_id].wait(-1) == 0:
            res = exe_net.requests[_request_id].outputs[out_blob]
            infered_images = infered_images + 1
            #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
            duration = time.time() - start_time
            print("inferred images: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)

# for test purpose only
image_number = 200

def preprocess_worker(image_queue, n, c, h, w):
    for i in range(1, 1 + image_number):
        image = cv2.imread("/opt/intel/computer_vision_sdk/deployment_tools/demo/car.png")
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        image_queue.put(image)
    image_queue.put(None)

start_time = -1

# ./async_api_multi-processes_multi-requests.py <request number>            
def main():
    global start_time

    # specify simutaneous request number in argv
    request_number = int(sys.argv[1])
    
    image_queue = multiprocessing.Queue(maxsize= request_number*3)

    model_dir = os.environ['HOME'] + "/model_downloader/object_detection/common/mobilenet-ssd/caffe/FP16/"
    model_xml = model_dir + "mobilenet-ssd.xml"
    model_bin = model_dir + "mobilenet-ssd.bin"
    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = plugin.load(network=net, num_requests=request_number*2)

    start_time = time.time()

    preprocess_thread = None
    preprocess_thread = threading.Thread(target=preprocess_worker, args=(image_queue, n, c, h, w))
    preprocess_thread.start()
    
    async_infer_worker(exec_net, request_number, image_queue, input_blob, out_blob)

    preprocess_thread.join()
    
    print()
    
    del exec_net
    del net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
