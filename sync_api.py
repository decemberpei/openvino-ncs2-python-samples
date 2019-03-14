from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import os
import time

model_dir = os.environ['HOME'] + "/model_downloader/object_detection/common/mobilenet-ssd/caffe/FP16/"
model_xml = model_dir + "mobilenet-ssd.xml"
model_bin = model_dir + "mobilenet-ssd.bin"
plugin = IEPlugin(device="MYRIAD")
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
exec_net = plugin.load(network=net, num_requests=1)

start_time = time.time()
image_number = 200
for i in range(1, 1 + image_number):
	image = cv2.imread("/opt/intel/computer_vision_sdk/deployment_tools/demo/car.png")
	image = cv2.resize(image, (w, h))
	image = image.transpose((2, 0, 1))
	image = image.reshape((n, c, h, w))
	res = exec_net.infer(inputs={input_blob: image}).get("detection_out")
	#print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
	duration = time.time() - start_time
	print("inferred frames: " + str(i) + ", average fps: " + str(i/duration) +"\r", end = '', flush = False)
print()

del exec_net
del net
del plugin
