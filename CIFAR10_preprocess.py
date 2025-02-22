import pickle
import tqdm
import numpy as np
from PIL import Image

def pickle_load(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def reshape(array):
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(array[:1024], (32, 32)) # Red channel
    image[...,1] = np.reshape(array[1024:2048], (32, 32)) # Green channel
    image[...,2] = np.reshape(array[2048:], (32, 32)) # Blue channel
    return image

raw_location = './datasets/CIFAR10/data/cifar-10-batches-py/'
save_location = './datasets/CIFAR10/images/'

train_1 = pickle_load(raw_location+'data_batch_1')
train_2 = pickle_load(raw_location + 'data_batch_2')
train_3 = pickle_load(raw_location + 'data_batch_3')
train_4 = pickle_load(raw_location + 'data_batch_4')
train_5 = pickle_load(raw_location + 'data_batch_5')

test = pickle_load(raw_location + 'test_batch')
meta = pickle_load(raw_location + 'batches.meta')

all_class = [label_name.decode("utf-8").replace("_", " ") for label_name in meta[b'label_names']]

class2images_train = {c: [] for c in all_class}

for train in [train_1, train_2, train_3, train_4, train_5]:
    for i in range(len(train[b'filenames'])):
        file_name = train[b'filenames'][i].decode("utf-8")
        label = train[b'labels'][i]
        array = train[b'data'][i]

        class_name = all_class[label]
        image = Image.fromarray(reshape(array))
        image.save(save_location + file_name)
        class2images_train[class_name].append(file_name)

class2images_test = {c: [] for c in all_class}
for i in range(len(test[b'filenames'])):
    file_name = test[b'filenames'][i].decode("utf-8")
    label = test[b'labels'][i]
    array = test[b'data'][i]

    class_name = all_class[label]
    image = Image.fromarray(reshape(array))
    image.save(save_location + file_name)
    class2images_test[class_name].append(file_name)

print(len(class2images_train['truck']), len(class2images_test['truck']))
#pickle.dump(class2images_train, open("class2images_train.p", "wb"))
#pickle.dump(class2images_test, open("class2images_test.p", "wb"))
