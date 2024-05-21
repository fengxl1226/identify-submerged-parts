import os
from PIL import Image
import sys
sys.path.append("./openpose/build_GPU/examples/tutorial_api_python")
import detect_keypoint_identify_part

def check(x_y, filename_segnet):
    seg_img = Image.open(filename_segnet)
    return any(seg_img.getpixel(p)[1] == 255 for p in x_y)

def dividend(points, num):
    points_list = []
    (x1, y1), (x2, y2) = points
    for i in range(num + 1):
        points_list.append((x1 + (x2-x1) * i / num, y1 + (y2-y1) * i / num))
    return points_list

def justice(folder_name):
    text_filename = folder_name + '/data.txt'
    fp = open(text_filename, "r", encoding="UTF-8")
    lines = fp.readlines()

    filename_segnet = folder_name + '/seg_img' + '/segnet.jpg'
    path = folder_name + '/clip_img'
    persons = os.listdir(path)
    persons = sorted(persons)

    submerged_part_filename = folder_name + '/submerged_part.txt'
    fw = open(submerged_part_filename, "w+", encoding="UTF-8")

    for l, line in enumerate(lines):

        line = line.replace("\n", "")
        line = line.replace("((", "(")
        line = line.replace("))", ")")
        line = line.replace(" ", "")
        str_list = line.split(",")
        Z = []
        for i in range(len(str_list)):
            if i % 2 == 0:
                Z.append(str_list[i] + "," + str_list[i+1])
        corners = []
        for z in Z:
            corners.append(eval(z))
        corners = dividend(points=corners, num=16)

        flag = check(corners, filename_segnet)
        if flag == True:
            print("The " + str(l+1) + "th person is in the water！")
            filename = os.path.join(path, persons[l])
            submerged_part_type = detect_keypoint_identify_part.get_pose(filename, str(l+1))
            fw.writelines(submerged_part_type + '\n')

        else:
            print("The " + str(l+1) + "th person is not in the water！")
            fw.writelines('\n')
            continue
    fw.close()