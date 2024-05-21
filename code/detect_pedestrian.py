import cv2
import os
from yolo_detector import Detector

def monister(filename):
    detector = Detector()
    base_name = os.path.basename(filename).split(".")[0]
    im = cv2.imread(filename)
    bboxes = detector.detect(im)
    num = len(bboxes)
    folder = "./database/" + os.path.basename(filename).split(".")[0]
    data_filename = folder + '/data.txt'
    fw = open(data_filename, "w+", encoding="UTF-8")

    clip_folder = folder + '/clip_img'

    for i in range(num):
        left = bboxes[i][0]
        top = bboxes[i][1]
        right = bboxes[i][2]
        bottom = bboxes[i][3]

        cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), 0)
        clip_im_file = base_name + "_" + str(i) + ".jpg"
        clip_im_file = clip_folder + "/" + clip_im_file
        im_clip = im[top:bottom, left:right]
        cv2.imwrite(clip_im_file, im_clip)

        coordinate = ((left, bottom), (right, bottom))

        fw.write(str(coordinate))
        fw.write("\n")
    fw.close()