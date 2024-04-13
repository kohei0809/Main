import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import pathlib
import shutil

from PIL import Image
from lavis.models import load_model_and_preprocess
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

from log_manager import LogManager
from log_writer import LogWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
lavis_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# Convolutional Neural Network Model
class CNNPhotoRemovalNetwork(nn.Module):
    def __init__(self):
        super(CNNPhotoRemovalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.contiguous().view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_data(folder_path, n_photos):
    image_folders = os.listdir(folder_path)
            
    image_folder_path = random.choice(image_folders)
    
    image_files = os.listdir(folder_path+"/"+image_folder_path)
    
    while True:
        images_path = random.sample(image_files, k=n_photos)
        n = 0
        for i in range(n_photos):
            if images_path[i][-1] == "g":
                n+=1
        if n == n_photos:
            break
    
    
    images = []
    image_path_list = []
    for i in range(n_photos):
        raw_image = Image.open(folder_path + "/" + image_folder_path + "/" + images_path[i])
        # numpy配列に変換
        image_array = np.array(raw_image)
        images.append(image_array)
        image_path_list.append(images_path[i])
        
    images = np.array(images)
    images = torch.from_numpy(images.astype(np.float32)).clone()
    images = images.permute(0,3,1,2)
    return images, image_folder_path, image_path_list

def get_target(image_folder_path, description_df, image_path_list):
    with open('PictureSelector/data/pictures/' + image_folder_path + '/_caption.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    ids = []
    captions = []
    for i in range(0, len(lines), 2):
        ids.append(lines[i].strip() + ".jpg")
        captions.append(lines[i+1].strip())

    caption_df = pd.DataFrame({'id': ids, 'caption': captions})
    
    captions = ""
    max_sim = -1.0
    max_idx = -1
    description = description_df[description_df["scene_id"]==image_folder_path]["description"].item()
    #print(f"image_folder: {image_folder_path}")
    for i in range(len(image_path_list)):
        captions = ""
        for j in range(len(image_path_list)):
            if i == j:
                continue
            #print(f"description: {description}")
            #print(f"image_path_list: {image_path_list[j]}")
            cap = caption_df[caption_df["id"]==image_path_list[j]]["caption"]
            #print(f"caption: {cap}")
            captions += caption_df[caption_df["id"]==image_path_list[j]]["caption"].item()
            captions += ". "
        similarity = calculate_similarity(captions, description)
            
        if similarity > max_sim:
            max_sim = similarity
            max_idx = i
        #print(f"captions: {captions}")
    
    return max_idx
            
def calculate_similarity(captions, origin_description):
    # 文をSentence Embeddingに変換
    embedding1 = bert_model.encode(captions, convert_to_tensor=True)
    embedding2 = bert_model.encode(origin_description, convert_to_tensor=True)
    
    # コサイン類似度を計算
    sentence_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return sentence_sim

def get_accuracy(output, target):
    if output == target:
        return 1
    else:
        return 0
    
def get_sim_out(image_folder_path, description_df, image_path_list, n_photos):
    # 削除する写真を決める
    # 他のdescriptionとの類似度を計算し、合計が最大のものを削除
    with open('PictureSelector/data/pictures/' + image_folder_path + '/_caption.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    ids = []
    captions = []
    for i in range(0, len(lines), 2):
        ids.append(lines[i].strip() + ".jpg")
        captions.append(lines[i+1].strip())

    caption_df = pd.DataFrame({'id': ids, 'caption': captions})
        
    sim_list = [[-10 for _ in range(n_photos)] for _ in range(n_photos)]
    for i in range(n_photos):
        for j in range(i+1, n_photos):
            cap1 = caption_df[caption_df["id"]==image_path_list[i]]["caption"].item()
            emd1 = bert_model.encode(cap1, convert_to_tensor=True)
            cap2 = caption_df[caption_df["id"]==image_path_list[j]]["caption"].item()
            emd2 = bert_model.encode(cap2, convert_to_tensor=True)
            sim_list[i][j] = util.pytorch_cos_sim(emd1, emd2).item()
            sim_list[j][i] = sim_list[i][j]
                
    total_sim = [sum(similarity_list) for similarity_list in sim_list]
    remove_index = total_sim.index(max(total_sim))
    return remove_index

def get_sub_list(data_path, image_folder_path, num):
    image_files = os.listdir(data_path+"/"+image_folder_path)
    
    while True:
        images_path = random.sample(image_files, k=num)
        n = 0
        for i in range(num):
            if images_path[i][-1] == "g":
                n+=1
        if n == num:
            break
    
    sub_list = []
    for i in range(num):
        sub_list.append(images_path[i])
        
    return sub_list


def get_sim_out2(image_folder_path, description_df, image_path_list, sub_path_list, n_photos, epoch=10):
    # 削除する写真を決める
    # 他のdescriptionとの類似度を計算し、合計が最大のものを削除
    with open('PictureSelector/data/pictures/' + image_folder_path + '/_caption.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    ids = []
    captions = []
    for i in range(0, len(lines), 2):
        ids.append(lines[i].strip() + ".jpg")
        captions.append(lines[i+1].strip())

    caption_df = pd.DataFrame({'id': ids, 'caption': captions})
        
    for k in range(epoch+1):
        sim_list = [[-10 for _ in range(n_photos)] for _ in range(n_photos)]
        for i in range(n_photos):
            for j in range(i+1, n_photos):
                cap1 = caption_df[caption_df["id"]==image_path_list[i]]["caption"].item()
                emd1 = bert_model.encode(cap1, convert_to_tensor=True)
                cap2 = caption_df[caption_df["id"]==image_path_list[j]]["caption"].item()
                emd2 = bert_model.encode(cap2, convert_to_tensor=True)
                sim_list[i][j] = util.pytorch_cos_sim(emd1, emd2).item()
                sim_list[j][i] = sim_list[i][j]
                    
        total_sim = [sum(similarity_list) for similarity_list in sim_list]
        remove_index = total_sim.index(max(total_sim))
        if k < epoch:
            image_path_list[remove_index] = sub_path_list[k]
    return remove_index, image_path_list

def get_random_out(n_photos):
    return random.randrange(n_photos)

# Training the CNN Photo Removal Network
def cal_accuracy(n_photos, n_epochs=10000, mode="random"):
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    #ログファイルの設定   
    log_manager = LogManager()
    log_manager.setLogDirectory("./PictureSelector/log/" + start_date)
    log_writer = log_manager.createLogWriter("check_accuracy")
    
    data_path = "PictureSelector/data/pictures"
    
    # データ作成
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
    
    accuracy = 0.0
    for epoch in range(1, n_epochs+1):
        data, image_folder_path, image_path_list = get_data(data_path, n_photos)
        if mode == "random":
            output = get_random_out(n_photos)
        if mode == "similarity":
            output = get_sim_out(image_folder_path, description_df, image_path_list, n_photos)
        if mode == "similarity2":
            sub_path_list = get_sub_list(data_path, image_folder_path, 10)
            output, image_path_list = get_sim_out2(image_folder_path, description_df, image_path_list, sub_path_list, n_photos, 10)
        
        target = get_target(image_folder_path, description_df, image_path_list)
        accuracy += get_accuracy(output, target)
        
        """
        if epoch <= 100:
            with open('out_pictures.txt', 'a') as f:
                ans = image_path_list[target]
                print(f"{epoch}: Folder: {image_folder_path} Index: {image_path_list} Target: {ans}", file=f)
            
            #フォルダがない場合は、作成
            source_path = "PictureSelector/data/pictures"
            target_path = "test_picture/" + str(epoch)
            p_dir = pathlib.Path(target_path)
            if not p_dir.exists():
                p_dir.mkdir(parents=True)
            for i in range(n_photos):
                shutil.copy(source_path+"/"+image_folder_path+"/"+image_path_list[i], target_path+"/"+image_path_list[i])
        """
        
        if epoch % 1000 == 0:
            ac = accuracy
            ac /= epoch
            with open('out_check_'+mode+'.txt', 'a') as f:
                time = datetime.datetime.now().strftime('%m-%d %H-%M-%S')
                print(f"{time}: epoch: {epoch}, Accuracy: {ac}", file=f)
                
    accuracy /= n_epochs
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    data_path = "PictureSelector/data" 
    print(data_path)
    
    n = 10  # Number of photos (n+1) for each input
    n_photos = n + 1
    #mode = "random"
    #mode = "similarity"
    mode = "similarity2"

    with open('out_check_'+mode+'.txt', 'a') as f:
        time = datetime.datetime.now().strftime('%m-%d %H-%M-%S')
        print(f"{time}: START {mode}", file=f)
    cal_accuracy(n_photos, n_epochs=10000, mode=mode)