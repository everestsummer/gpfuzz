import numpy as np
import os

#暂时限制文件最大长度128x128.(后面也可以试试256x256)
def read_file(filepath, H=128, W=128):  
  max_len = H * W
  fd = open(filepath, "rb")
  size = os.path.getsize(filepath)
  if size > max_len:
    size = max_len
  bin_data = fd.read(size)
  fd.close()

  #用0做padding。
  data_arr = [0 for i in range(max_len)]
  for i in range(size):
    data_arr[i] = bin_data[i]

  np_arr = np.array(data_arr)
  np_arr = np_arr.reshape(H, W)

  return np_arr