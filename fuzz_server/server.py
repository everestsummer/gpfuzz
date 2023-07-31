import socket, os, glob
import _thread
import time
import shutil
import random
import subprocess

def DEBUG(data):
  if True:
    print(data)


def send_seed(classid, put_seed_to):
  seed_pool = "/mnt/e/SMARTFuzzer/seedpool/" + str(int(classid)) #Prevent folder traversal attack from client
  if not os.path.isdir(seed_pool):
    print("Error: ", seed_pool, " is not on the server!!")
    return -1
  seed_sample = random.sample(os.listdir(seed_pool), 50) #random 50 files

  try:
    os.mkdir(os.path.abspath(put_seed_to))
  except:
    pass

  for fn in seed_sample:
    srcpath = os.path.join(seed_pool, fn)
    #DEBUG("Sending seed file from " + srcpath + " to " + os.path.abspath(os.path.join(put_seed_to, fn)))
    #as we may have "../reload" in the path and this may make shutil.copyfile return file not found error
    shutil.copyfile(srcpath, os.path.abspath(os.path.join(put_seed_to, fn)))
  return 0


def on_new_client(clientsocket, addr):
  while True:
    msg_ = clientsocket.recv(20480)
    if not msg_:
      DEBUG("ERROR: Client socket recv() failed, disconnecting.")
      break

    DEBUG("INFO: New client request incoming.")
    msg = msg_.decode("utf-8")
    DEBUG("msg: " + str(msg))
    msg_arr = msg.split("\n")
    if msg_arr[0] == "request_identify":
      DEBUG("PARSER: Got request_identify")
      if msg_arr[1][:9] == "filelist:":
        DEBUG("PARSER: Got filelist")
        filelist = msg_arr[1][9:].split("|")
        if msg_arr[2][:5] == "from:" and msg_arr[3][:9] == "put_into:":
          DEBUG("PARSER: Got from and put_into")
          from_client = msg_arr[2][5:]
          put_seed_to = msg_arr[3][9:]
          start_time = str(int(time.time()))
          folder_name = "fuzz_data_" + from_client + "/input/" + from_client + "/" + start_time

          DEBUG("INFO: Inference request received, preparing data at " + folder_name)

          #Create folder, if we already have one, delete all files from the old folder.
          try:
            os.makedirs(folder_name)
          except FileExistsError: 
            #Not likely to happen
            DEBUG("INFO: Delete all files in " + folder_name + " because we already have something under that folder.")
            files = glob.glob(folder_name + "/*")
            for f in files:
              os.remove(f)
            pass

          #Copy remote file to local folder, and generate file list correspondingly.
          DEBUG("INFO: Copying remote files.")
          i = 0
          gen_filenamelist = ""
          for fn in filelist:
            if fn == "" or not os.path.exists(fn):
              DEBUG("WARNING: empty file name or non-exist file: " + fn)
              continue
            #DEBUG("INFO: Copying from " + fn + " to " + os.path.join(folder_name, str(i)))
            shutil.copy(fn, os.path.join(folder_name, str(i)))
            gen_filenamelist += "input/" + from_client + "/" + start_time + "/" + str(i) + "\n"
            i += 1

          DEBUG("INFO: Copy done, generate filelist.txt file.")
          #Generate file name list, for inferencing.
          gen_filenamelist_file = "fuzz_data_" + from_client + "/filelist.txt"  #Fixed name, dont change
          with open(gen_filenamelist_file, "w+") as f:
            f.write(gen_filenamelist)

          DEBUG("INFO: Starting model to classfy those files. It may take minutes.")
          #Start model and classify the input.
          #For legacy python support check:
          #https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output
          gpmodel_result = subprocess.check_output(['python3', './classifier7.py', '--fuzz_data_dir', "fuzz_data_" + from_client])

          gpmodel_result = str(gpmodel_result)

          #DEBUG("DEBUG: Model returns: " + gpmodel_result)

          #parse the ret value from the model, extrace the class number.
          #do nothing if we can't find "Predict class: " from the output.
          gpmodel_result = str(gpmodel_result)
          predict_idx = gpmodel_result.find("Predicted class: ")
          if predict_idx == -1:
            print("Error: ", gpmodel_result)
            continue #do nothing if we didn't find any result.
          else:
            newline_pos = gpmodel_result.find("\n", predict_idx)
            if newline_pos == -1:
              newline_pos = len(gpmodel_result)
            predict_result = gpmodel_result[predict_idx + len("Predicted class: ") : newline_pos]
            predict_result = predict_result.strip() #remove unnecessary bytes.
            DEBUG("INFO: Parsed result = " + predict_result)
            predict_result_fixed = ""
            for ch in predict_result:
              DEBUG("DEBUG: Scannig char: " + ch)
              if ch == '-' or ch.isnumeric():
                predict_result_fixed += ch
              else:
                break
            int_result = int(predict_result_fixed)
            DEBUG("INFO: Parsed class = " + predict_result_fixed)
          #Send files according to the return value.
          #First we locate the seedpool, sample 50 random files from that pool
          #Then we copy them to the folder where client ask us to put into.
          if int_result != -1:
            send_seed(int_result, put_seed_to)
          else:
            DEBUG("INFO: Don't know what the class is, skipping sending seed")

          #Send message to client, notify it that we've got a class number for it.
          send_result = send_message_to(from_client, "identify_result\nclass_id:" + str(int_result) + "\n")
          if send_result == -1:
            print("Error sending message to " + from_client)

    #################################
    elif msg_arr[0] == "request_seed":
      if msg_arr[1][:9] == "class_id:":
        if msg_arr[2][:9] == "put_into:":
          put_seed_to = msg_arr[2][9:]
          class_id = int(msg_arr[1][9:])
          DEBUG("INFO: Direct put request received, clsid: " + msg_arr[1][9:])
          if class_id >= 0:
            send_seed(class_id, put_seed_to)
    else:
      print("ERROR: unsupported incoming request: ", msg)

  clientsocket.close()

def send_message_to(who, what):
  ss = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  ss.connect("/tmp/client_" + str(who) + ".socket2")
  if not ss:
    DEBUG("ERROR: Cannot connect to client /tmp/client_"  + str(who) + ".socket2")
    return -1
  # remember to use raw_input while send string to client.
  msg = str.encode(what)
  ss.send(msg)
  DEBUG("INFO: message sent: \n" + str(what))
  ss.close()
  return 0



s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

try:
  os.remove("/tmp/gpfuzz.socket")
except OSError:
  DEBUG("WARNING: /tmp/gpfuzz.socket exists and can not be removed!")
  pass


DEBUG("INFO: Starting, listening at /tmp/gpfuzz.socket.")

s.bind("/tmp/gpfuzz.socket")
s.listen(1)

while True:
   c, addr = s.accept()
   DEBUG("INFO: A new client has connected to us.")
   _thread.start_new_thread(on_new_client,(c,addr))

s.close()
