import os
import segment_watersurface
import detect_pedestrian
import check_pedestrian_in_water

def create_folder(folder_path):
    for filename in os.listdir(folder_path):
        folder_name = os.path.basename(filename).split(".")[0]
        folder_name = "./database/" + folder_name
        folders = ['clip_img', 'seg_img', 'keypoints']

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for folder in folders:
            folder_path = os.path.join(folder_name, folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

def identify_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        segment_watersurface.predict_water(file_path)
        detect_pedestrian.monister(file_path)
        folder_name = os.path.basename(file_path).split(".")[0]
        folder_name = "./database/" + folder_name
        check_pedestrian_in_water.justice(folder_name)

if __name__ == '__main__':
    folder_path = './img'
    create_folder(folder_path)
    identify_folder(folder_path)


