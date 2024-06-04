import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dir_path = "data/scene_datasets/mp3d"
dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

# ファイルを読み込んで行ごとにリストに格納する
with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
    lines = file.readlines()

# scene id と文章を抽出してデータフレームに変換する
scene_ids = []
descriptions = []
for i in range(0, len(lines), 3):
    scene_ids.append(lines[i].strip())
    descriptions.append(lines[i+2].strip())

description_df = pd.DataFrame({'scene_id': scene_ids, 'description': descriptions})

for i in range(90):
    scene_name = dirs[i]
    print(str(i) + ", START FOR: " + scene_name)

    description = description_df[description_df["scene_id"]==scene_name]["description"].item()
    if description == "wip":
        continue

    df = pd.read_csv("research_picture_value_and_similarity/csv/similarity_" + scene_name + ".csv", names=['similarity', 'value'], header=None)
    
    plt.scatter(df['value'], df['similarity'])
    # ラベルの追加
    plt.xlabel('Value of Picture')
    plt.ylabel('Similarity')    
    # save graph
    plt.savefig('./research_picture_value_and_similarity/scatter/' + scene_name + '.png')    
    plt.clf()

print("Showing Saliency and Similarity Scatter is completed.")