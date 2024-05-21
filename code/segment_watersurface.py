import copy
import os
import numpy as np
from PIL import Image
from nets.segnet import convnet_segnet

def predict_water(filename):
    class_colors = [[0, 0, 0], [0, 255, 0]]

    #   Image height and width
    HEIGHT = 416
    WIDTH = 416

    #   category: background + water = 2
    NCLASSES = 2

    #   Load model
    model = convnet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)

    #   model_path: Our trained weights
    model_path = "./weights/ep049-loss0.020-val_loss0.128.h5"
    model.load_weights(model_path)

    img = Image.open(filename)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH, HEIGHT), Image.BICUBIC)
    img = np.array(img) / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)

    pr = model.predict(img)[0]
    pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c) * class_colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * class_colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * class_colors[c][2]).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    folder = "./database/" + os.path.basename(filename).split(".")[0]
    folder = folder + '/seg_img'
    filename_segnet = folder + '/segnet.jpg'
    seg_img.save(filename_segnet)

    image = Image.blend(old_img, seg_img, 0.3)

    filename_segnet_blend = folder + "/segnet_blend.jpg"
    image.save(filename_segnet_blend)









