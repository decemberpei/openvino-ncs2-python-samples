# beblow is the test result on an Intel i5-7200U ultrabook
config|Parameters|FPS
---|:--:|---:
sync_api.py|N/A|16.050873524801453
async_api.py|N/A|29.352513957879324
async_api_multi-threads.py|N/A|39.667721322205055
async_api_multi-processes.py|N/A|41.084748784333684
async_api_multi-threads_multi-requests.py|2|41.854912389218885
async_api_multi-threads_multi-requests.py|4|46.095173369691004
async_api_multi-processes_multi-requests.py|2|43.477770517034926
async_api_multi-processes_multi-requests.py|4|46.070958743063564
async_api_multi-threads_multi-requests_multi-ncs.py|2 1|60.313427342506766
async_api_multi-threads_multi-requests_multi-ncs.py|2 2|79.82939749065703
async_api_multi-threads_multi-requests_multi-ncs.py|2 4|87.53705811233776
async_api_multi-processes_multi-requests_multi-ncs.py|2 1|61.018100651765456
async_api_multi-processes_multi-requests_multi-ncs.py|2 2|86.02675250395884
async_api_multi-processes_multi-requests_multi-ncs.py|2 4|89.02194282781669

# also the test result on UP squared (UP2) dev board (Intel® Celeron™ N3350) with standard Ubuntu 16.04
config|Parameters|FPS
---|:--:|---:
sync_api.py|N/A|16.534288935980122
async_api.py|N/A|30.047453123836432
async_api_multi-threads.py|N/A|36.182525107032216
async_api_multi-processes.py|N/A|38.529989918532984
async_api_multi-threads_multi-requests.py|2|43.388968704276266
async_api_multi-threads_multi-requests.py|4|43.513101484307526
async_api_multi-processes_multi-requests.py|2|43.061900126219516
async_api_multi-processes_multi-requests.py|4|45.553271118272936
async_api_multi-threads_multi-requests_multi-ncs.py|2 1|58.006486187463274
async_api_multi-threads_multi-requests_multi-ncs.py|2 2|74.36024288188348
async_api_multi-threads_multi-requests_multi-ncs.py|2 4|78.56850835647774
async_api_multi-processes_multi-requests_multi-ncs.py|2 1|55.739910963474496
async_api_multi-processes_multi-requests_multi-ncs.py|2 2|77.52199693575577
async_api_multi-processes_multi-requests_multi-ncs.py|2 4|73.48973036182308
