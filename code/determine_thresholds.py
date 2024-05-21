import pandas as pd
import os
import numpy as np

# Initialize node and index lists
order = [14, 11, 13, 10, 12, 8, 9, 5, 2, 1, 0, 18, 17, 16, 15]  # 15 nodes 
index = ["nose", "neck", "right shoulder", "right elbow", "right wrist", "left shoulder", "left elbow", "left wrist",
         "middle hip", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle", "right eye",
         "right ear", "left ear", "left thumb", "left little finger", "left heel", "right thumb", "right little finger",
         "right heel"]  # Names of nodes 

myorder_list = [index[i] for i in order]

# Set data path and read ground truth values file
database_path = 'database'
folder_list = [folder_name for folder_name in os.listdir(database_path)]
column_list = ['Foot', 'Calf', 'Thigh', 'Waist or chest', 'Neck or head', 'None']

True_submerged = pd.read_csv('1.csv', encoding='gbk', index_col=0)

# Initialize submerged part dictionary
def initialize_submerged_dict():
    return {x: {y: 0 for y in column_list} for x in column_list}

person_submerged_part_dict_2D = initialize_submerged_dict()

# Single search function
def grid_search_once(C1, C2, C3, C4):
    global person_submerged_part_dict_2D
    person_submerged_part_dict_2D = initialize_submerged_dict()
    person_list = []
    count = 0

    for folder in folder_list:
        filenames = os.listdir(os.path.join(database_path, folder, 'keypoints'))
        for filename in filenames:
            if filename.endswith('.csv'):
                count += 1
                file_path = os.path.join(database_path, folder, 'keypoints', filename)
                person_name = f"{folder}_{int(filename.split('_')[-1][:-4]) - 1}"
                person_list.append(person_name)
                df = pd.read_csv(file_path, index_col='Name')

                for order_name in myorder_list:
                    if df.loc[order_name]['X'] != 0 and df.loc[order_name]['Y'] != 0 and df.loc[order_name]['Confidence'] != 0:
                        confidence = df.loc[order_name]['Confidence']
                        true_submerged = True_submerged.loc[person_name]['Ground Truth']

                        if (order_name in ["left ankle", "right ankle"] and confidence > C1) or \
                           (order_name in ["left knee", "right knee"] and confidence > C2) or \
                           (order_name in ["left hip", "right hip", "middle hip"] and confidence > C3) or \
                           (order_name in ["left shoulder", "right shoulder"] and confidence > C4):
                            person_submerged_part_dict_2D[column_name(order_name)][true_submerged] += 1
                            break
                        elif confidence > 0:
                            if order_name in ["nose", "left eye", "right eye", "left ear", "right ear", "neck"]:
                                person_submerged_part_dict_2D["Neck or head"][true_submerged] += 1
                            else:
                                person_submerged_part_dict_2D["None"][true_submerged] += 1
                            break

    accuracy = sum(person_submerged_part_dict_2D[c][c] for c in column_list) / count
    return accuracy, pd.DataFrame(person_submerged_part_dict_2D)

# Function to get column name based on order_name
def column_name(order_name):
    if order_name in ["left ankle", "right ankle"]:
        return "Foot"
    elif order_name in ["left knee", "right knee"]:
        return "Calf"
    elif order_name in ["left hip", "right hip", "middle hip"]:
        return "Thigh"
    elif order_name in ["left shoulder", "right shoulder"]:
        return "Waist or chest"
    else:
        return "Neck or head"

# Define grid search ranges
c1_list = np.linspace(0.35, 0.45, num=11)
c2_list = np.linspace(0.10, 0.35, num=26)
c3_list = np.linspace(0.05, 0.15, num=11)
c4_list = np.linspace(0.05, 0.15, num=11)

# Store search results
c1_c2_list = []
accuracy_list = []
df_list = []

for c1 in c1_list:
    for c2 in c2_list:
        for c3 in c3_list:
            for c4 in c4_list:
                accuracy, my_df = grid_search_once(c1, c2, c3, c4)
                c1_c2_list.append([c1, c2, c3, c4])
                accuracy_list.append(accuracy)
                df_list.append(my_df)

total_dict = {
    "(c1, c2, c3, c4)": c1_c2_list,
    "accuracy": accuracy_list,
    "dataframe": df_list
}

total_df = pd.DataFrame(total_dict)
total_df.sort_values("accuracy", ascending=False, inplace=True)

n = 10  # Output the top n results with the best accuracy

if not os.path.exists("output"):
    os.makedirs("output")

for i in range(n):
    total_df.iloc[i]["dataframe"].to_csv(f"output/rank_{i}.csv", encoding='utf_8_sig')

total_df.drop('dataframe', axis=1, inplace=True)
total_df.to_csv('rank.csv', encoding='utf_8_sig')
