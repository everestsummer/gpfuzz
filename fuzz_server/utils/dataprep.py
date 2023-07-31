# data preparation utils

import numpy as np
import tensorflow as tf


def partitionByClass(X, y_true):
    maxc = np.max(y_true + 1)
    ids = [[] for i in range(maxc)]
    for i in range(np.shape(y_true)[0]):
        ids[y_true[i]].append(i)
    return ids

def prepare_batch_inference(X_support, y_support, X_query, y_query, ids_by_class, N_classes = 12, N_support_per_class = 10, N_query = 50, query_start_idx = 0, permute=True):
    max_class_number = np.max(y_support)  # max class number of support set, as we know nothing about y_query, assume it to be one of this class.
    # 这么写是正确的，它会生成一个包含Nclasses个元素的列表，列表各项是打乱顺序的0-max_class_number，而且不重复。
    class_shuffled = np.random.choice(range(max_class_number + 1), size=(N_classes), replace=False)  # choose subset of N_classes classes

    ids_support = np.array(
        [np.random.choice(ids_by_class[c], size=(N_support_per_class), replace=False) for c in class_shuffled]
    )

    # a dummy array, stores all 0 (it doesn't affect the result, just a placeholder)
    ids_query = np.array(
        [0 for i in range(N_query)]
    )

    ids_batch_support = np.ndarray.flatten(ids_support)
    ids_batch_query = np.ndarray.flatten(ids_query)


    if permute:
        ids_batch_support = np.random.permutation(ids_batch_support)
        #ids_batch_query = np.random.permutation(ids_batch_query)  #Meaningless.

    return X_support[ids_batch_support, :], y_support[ids_batch_support],  \
           X_query[N_query * query_start_idx: N_query * (query_start_idx + 1), :], y_query[ids_batch_query], \
           class_shuffled


def prepare_batch_no_y(X, ids_by_class_train, N_support=10, N_query=5, permute=True):
    #我们一次只会传一个类别进来，因此这里直接设置成一类，号码为0
    classes = [0]

    #因此这里相当于在ids_by_class_train[0]里面随机取support+query个样本来作为batch
    ids_batch = np.array(
        [np.random.choice(ids_by_class_train[c], size=(N_support + N_query), replace=False) for c in classes]
    )

    #然后切割开
    ids_batch_support = np.ndarray.flatten(ids_batch[:, :N_support])
    ids_batch_query = np.ndarray.flatten(ids_batch[:, N_support:])

    if permute:
        ids_batch_support = np.random.permutation(ids_batch_support)
        ids_batch_query = np.random.permutation(ids_batch_query)

    #并且返回
    return X[ids_batch_support, :], X[ids_batch_query, :], classes


def prepareBatch(X, y_true, ids_by_class_train, N_classes=10, N_support=10, N_query=5, permute=True):
    maxc = np.max(y_true)  # max class number

    classes = np.random.choice(range(maxc + 1), size=(N_classes), replace=False)  # choose subset of N_classes classes

    #len_arr = [len(ids_by_class_train[c]) for c in classes]
    #print("Items count by each classes:", len_arr)

    #遍历每个类，然后在每个类里都随机抽support+query个。 这个数据源就是ids_by_class_train。
    ids_batch = np.array(
        [np.random.choice(ids_by_class_train[c], size=(N_support + N_query), replace=False) for c in classes]
    )

    ids_batch_support = np.ndarray.flatten(ids_batch[:, :N_support])
    ids_batch_query = np.ndarray.flatten(ids_batch[:, N_support:])

    if permute:
        ids_batch_support = np.random.permutation(ids_batch_support)
        ids_batch_query = np.random.permutation(ids_batch_query)

    return X[ids_batch_support, :], y_true[ids_batch_support], X[ids_batch_query, :], y_true[
        ids_batch_query], classes


# preprocessing images (loaded background 1.0, character 0.0)
def invert_img(x):
    _, H, W = np.shape(x)
    return -2.0 * np.reshape(x, [-1, H, W]) + 1.0


def deinvert_img(x):
    _, H, W = np.shape(x)
    return 1.0 - 0.5 * x


def resize_img(x, Hold, Wold, Hnew, Wnew):
    q = tf.compat.v1.Session().run(tf.image.resize(tf.reshape(x, [-1, Hold, Wold, 1]), [Hnew, Wnew]))
    return np.reshape(q, [-1, Hnew, Wnew])


def subtract_mean(X):
    #N, SIZE = np.shape(X)
    #Xf = np.reshape(X, [N, SIZE])
    Xf = X
    means = np.mean(Xf, axis=1, keepdims=True)
    Xf = Xf - np.mean(Xf, axis=1, keepdims=True)
    #return np.reshape(Xf, np.shape(X)), means
    return Xf, means

def augment_by_rotations(X, y, ks=[0, 1, 2, 3]):
    Xs, ys = [], []
    class_step = np.max(y) + 1
    for i, k in enumerate(ks):
        Xs.append(np.rot90(X, k=k, axes=(1, 2)))
        ys.append(np.array(y) + (i) * class_step)
    Xa = np.concatenate(Xs, axis=0)
    ya = np.concatenate(ys, axis=0)
    return Xa, ya
