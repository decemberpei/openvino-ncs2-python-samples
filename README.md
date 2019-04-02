# Openvino IE(Inference Engine) python samples - NCS2
## before you start, make sure you have
1. Dev machine with Intel 6th or above Core CPU (Ubuntu is preferred, a Win 10 should also work)
2. Openvino 2018R5 or later installed and configured for NCS devices
3. physical NCS2 VPU (the first gen NCS should also work, with a much lower perf)
4. mobilenet-ssd caffe model downloaded, IR model file generated with MO ( [what is IR/MO?](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer))

## how to run (remember to `source /opt/intel/computer_vision_sdk/bin/setup_vars.sh` in each new terminal)
1. the basic "hello world" sample using IE sync API (you get a FPS around 16FPS on a i5-7200u in terminal outputs):

`python3 sync_api.py`
2. the async sample using IE async API (this will boost you to 29FPS on a i5-7200u):

`python3 async_api.py`
3. the 'async API' + 'multiple threads' implementation (this will boost you to 39FPS on a i5-7200u):

`python3 async_api_multi-threads.py`
4. the 'async API' + 'multiple threads' + 'simutaneous inference requests' implementation (this will boost you to 46FPS on a i5-7200u if $requests_number is 4):

`python3 async_api_multi-threads_multi-requests.py [requests_number]`
5. the 'async API' + 'multiple threads' + 'simutaneous inference requests' + 'multiple ncs' implementation (this will boost you to 87FPS on a i5-7200u if $ncs_number is 2 and $requests_number is 4)

`python3 async_api_multi-threads_multi-requests_multi-ncs.py [ncs_number] [requests_number]`
5. the 'async API' + 'multiple processes' + 'simutaneous inference requests' + 'multiple ncs' implementation (this will boost you to 89FPS on a i5-7200u if $ncs_number is 2 and $requests_number is 4)

`python3 async_api_multi-processes_multi-requests_multi-ncs.py [ncs_number] [requests_number]`
