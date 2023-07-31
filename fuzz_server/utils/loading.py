# library for Omniglot data loading and preprocessing

import matplotlib.pyplot as plt
import numpy as np

import utils.visualize
import utils.file_reader

import os

def loadList(path, list_name, limit=None, verbose=False):
    class_info = []
    class_ids = []
    images = []

    print("loadlist entered!")
    class_id_now = -1
    class_name_now = ""

    p = list_name
    list_path = os.path.join(path, p)

    f = open(list_path, "r")
    lines = f.readlines()

    #change limit as "limit files per folder."
    #if limit is not None:
    #    lines = lines[:limit]

    idx_in_folder = 0
    if limit is None:
        limit = 9999999999
    print("L=", limit)

    for line in lines:  # goes through each character example
        image_path = line[:-1]
        subset_name, alphabet_name, character_name, image_name = image_path.split("/")

        #把完整类别信息保存起来。
        if idx_in_folder < limit:
            class_info.append((subset_name, alphabet_name, character_name))

        #这里的处理方式显然是——预期那个list里面是按文件名顺序排序的，
        #前一个“类别”的图片在列表里是连在一起的。
        #
        #一个文件夹（大类+小类别，例如“英语1”和“英语2”）内的所有数据处理完
        #就给类别号码+1，用于区分。
        #
        #不过，这只是对大类的子类进行区分。
        #例如“英语1”和“日语1”、“俄语1”，都会记为类别0；
        #   “英语2”和“日语2”、“俄语2”，都会记为类别1……

        #print(alphabet_name + character_name)
        if alphabet_name + character_name != class_name_now:  # new class encoutered
            #Redo the step above.
            if idx_in_folder >= limit:
                class_info.append((subset_name, alphabet_name, character_name))
            idx_in_folder = 0 #Reset counter
            class_id_now += 1
            class_name_now = alphabet_name + character_name
            print("New class id ", class_id_now, " for class name: ", class_name_now)

        #Set limits on it.
        if idx_in_folder >= limit:
            continue

        class_ids.append(class_id_now)

        img_filename = os.path.join(path, image_path)

        if verbose:
            print(img_filename)

        ###############################################
        #TODO:
        #读取图像，这里我们可以改成读取文本了
        ###############################################
        im = utils.file_reader.read_file(img_filename) #plt.imread(img_filename, format='png')

        #im = np.array(im)

        # print(np.mean(im))
        # print(im.shape)

        # plt.imshow(im)
        # plt.show()

        images.append(im)
        idx_in_folder += 1


    images = np.array(images)
    class_ids = np.array(class_ids)

    # print(images.shape)

    return class_ids, images, class_info


#其实这两个函数是一样的。。只是故意叫不同名字区分开吧
# 1. 从训练集里面读取数据，
def load_existing_train_data(path, limit=None):
    labels, files, info = loadList(path, "train.txt", verbose=False, limit=limit)
    return labels, files, info

# 2. 从Fuzzer输入里面读取数据，（label unknown，全部设为0）
def load_fuzzer_input(path, limit=None):
    #Labels has no use but only keep it here, so we don't need to create another empty array.
    # you can use labels *= 0 to clear it if you want.
    labels, files, info = loadList(path, "filelist.txt", verbose=False, limit=limit)
    return labels, files, info

def loadOmniglot(path="../data/", train=0, train_limit=None, val_limit=None):
    #lists = [
    #    "train.txt",
    #    "val.txt",
    #]

    labels_train, images_train, info_train = loadList(path, "train.txt", verbose=False, limit=train_limit)
    labels_val, images_val, info_val = loadList(path, "val.txt", verbose=False, limit=val_limit)

    return (labels_train, images_train, info_train, labels_val, images_val, info_val)



#下面是测试用的代码，不用管。
# for testing, if ran directly
if __name__ == "__main__":
    labels_train, images_train, info_train, labels_val, images_val, info_val = loadOmniglot(train=0, limit=None)

    ids = range(images_train.shape[0])
    ids = [i * 20 for i in range(int(images_train.shape[0] / 20))]  # just single copy of a character

    utils.visualize.visualize(
        images_train[ids, :, :],
        output="characters_train_small.png",
        width=25
    )

    ids = range(images_val.shape[0])
    ids = [i * 20 for i in range(int(images_val.shape[0] / 20))]  # just single copy of a character

    utils.visualize.visualize(
        images_val[ids, :, :],
        output="characters_val_small.png",
        width=25
    )
