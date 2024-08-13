import sys
import cv2
import os
from sys import platform
import argparse
import csv

order = [14, 11, 13, 10, 12, 8, 9, 5, 2, 1, 0, 18, 17, 16, 15]  # 15 key points

index = ["nose", "neck", "right shoulder", "right elbow", "right wrist", "left shoulder", "left elbow", "left wrist",
             "middle hip", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle", "right eye", "left eye",
             "right ear", "left ear", "left thumb", "left little finger", "left heel", "right thumb", "right little finger", "right heel"]

def get_pose(filename, name):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('./openpose/build_GPU/python/openpose/Release');
                os.environ['PATH'] = os.environ['PATH'] + ';' + './openpose/build_GPU/x64/Release;' + './openpose/build_GPU/bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('./openpose/build_GPU/python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()

        parser.add_argument("--image_path", default=filename,
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")

        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "./openpose/models/"
        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        folder1 = './database/' + os.path.basename(filename).split(".")[0]
        folder_path = folder1 + '/clip_img'
        cv2.imwrite(os.path.join(folder_path, args[0].image_path), datum.cvOutputData)
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)

        folder2 = './database/' + os.path.basename(filename).split("_")[0]
        folder2 = folder2 + '/keypoints'
        csv_file_path = folder2 + '/keypoints_' + name + '.csv'

        poseKeypoints = datum.poseKeypoints[0]
        list_tuple_keypoint = []
        submerged_part = ""
        for k in order:
            keypoint = poseKeypoints[k]
            x = float(keypoint[0])
            y = float(keypoint[1])
            c = float(keypoint[2])
            name = index[k]
            tuple_keypoint = (name, x, y, c)
            list_tuple_keypoint.append(tuple_keypoint)

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Name', 'X', 'Y', 'Confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for tuple_keypoint in list_tuple_keypoint:
                writer.writerow({'Name': tuple_keypoint[0], 'X': tuple_keypoint[1], 'Y': tuple_keypoint[2],
                                 'Confidence': tuple_keypoint[3]})

        for tuple_keypoint in list_tuple_keypoint:
            c = tuple_keypoint[3]
            name = tuple_keypoint[0]
            if name == "left ankle" or name == "right ankle":
                if c > 0.42:
                    submerged_part = "Foot"
                    break
            elif name == "left knee" or name == "right knee":
                if c > 0.11:
                    submerged_part = "Calf"
                    break
            elif name == "left hip" or name == "right hip" or name == "middle hip":
                if c > 0.09:
                    submerged_part = "Thigh"
                    break
            elif name == "left shoulder" or name == "right shoulder":
                if c > 0.09:
                    submerged_part = "Waist or chest"
                    break
            else:
                if c > 0:
                    if name == "nose" or name == "left eye" or name == "right eye" or name == "left ear" or name == "right ear" or name == "neck":
                        submerged_part = "Neck or head"
                        break
                    else:
                        submerged_part = "None"
                        break
        print(submerged_part)
        return submerged_part

    except Exception as e:
        print(e)
        return "None"