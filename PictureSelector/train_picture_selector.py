import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import pathlib

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
        self.N = 1
        self.conv1 = nn.Conv2d(3*self.N, 16*self.N, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16*self.N, 32*self.N, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32*self.N, 64*self.N, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32 * self.N, 256*self.N)
        self.fc2 = nn.Linear(256*self.N, 1*self.N)

    def forward(self, x):
        print(x.shape)
        print(x)
        a = torch.cat([x[i] for i in range(self.N)], dim=1)
        with open('output3.txt', 'a') as f:
            print(a, file=f)
        print(a.shape)
        print(a)
        x = a
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.contiguous().view(-1, 64 * 32 * 32 * self.N)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_data(folder_path, n_photos):
    image_folders = os.listdir(folder_path)
            
    image_folder_path = random.choice(image_folders)
    #print(f"image_folder_path: {image_folder_path}")
    
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
    """
    target = []
    for i in range(len(image_path_list)):
        if i == max_idx:
            target.append(1.0)
        else:
            target.append(0.0)
            
    target = torch.from_numpy(np.array(target).astype(np.float32)).clone()
    """
    target = torch.tensor([max_idx], dtype=torch.int64)
    return target
            
def calculate_similarity(captions, origin_description):
    #print(f"captions: {captions}")
    #print(f"origin: {origin_description}")
    # 文をSentence Embeddingに変換
    embedding1 = bert_model.encode(captions, convert_to_tensor=True)
    embedding2 = bert_model.encode(origin_description, convert_to_tensor=True)
    
    # コサイン類似度を計算
    sentence_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return sentence_sim

def cal_accuracy(output, target):
    max_output_idx = torch.argmax(output)
    max_target_idx = target.item()
    if max_output_idx == max_target_idx:
        return 1
    else:
        return 0

# Training the CNN Photo Removal Network
def train_cnn_photo_removal_network(n_photos, n_epochs=1000000, lr=0.001, save_path='model.pth'):
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    with open('out_print3.txt', 'a') as f:
        print(f"START: {start_date}", file=f)
    #ログファイルの設定   
    log_manager = LogManager()
    log_manager.setLogDirectory("./PictureSelector/log/" + start_date)
    log_writer = log_manager.createLogWriter("train_removal_network")
    
    model = CNNPhotoRemovalNetwork().to(device)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
        #print(data.shape)
        optimizer.zero_grad()
        output = model(data.to(device))
        #print(f"output: {output}")
        target = get_target(image_folder_path, description_df, image_path_list).to(device)
        #print(f"target: {target}")
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        accuracy += cal_accuracy(output, target)

        if epoch % 1000 == 0:
            accuracy /= 1000
            log_writer.writeLine(str(epoch) + "," + str(loss.item()) + "," + str(accuracy))
            #print(f"epoch: {epoch}")
            time = datetime.datetime.now().strftime('%m-%d %H-%M-%S')
            with open('out_print3.txt', 'a') as f:
                print(f'{time}: Epoch {epoch}/{n_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}', file=f)
            accuracy = 0.0
            
            with open('output3.txt', 'a') as f:
                print(f'{time}', file=f)
                print(f"output: {output}", file=f)
                print(f"target: {target}", file=f)
                print(f"loss: {loss.item()}", file=f)
                print(f"acc: {cal_accuracy(output, target)}", file=f)
            
        if epoch % 10000 == 0:
            # Save the trained model
            torch.save(model.state_dict(), save_path+ "_" + str(epoch) + ".pth")
            print(f'Model saved to {save_path}')

    return model

# Function to load a saved model
def load_cnn_photo_removal_network(model_path):
    model = CNNPhotoRemovalNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f'Model loaded from {model_path}')
    return model

def captioning(data_path):    
    # setup device to use
    device = (
        torch.device("cuda", 1)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    lavis_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    folder_name = "pictures"
    folder_path = os.path.join(data_path, folder_name)
    image_folders = os.listdir(folder_path)
        
    #フォルダがない場合は、作成
    p_dir = pathlib.Path(folder_path)
    if not p_dir.exists():
        p_dir.mkdir(parents=True) 
            
    print(f"folder_path: {folder_path}")
    for j in range(len(image_folders)):
        image_folder_path = image_folders[j]
        print(f"{j}, image_folder_path: {image_folder_path}")
        if os.path.isfile(folder_path + "/" + image_folder_path + '/_caption.txt'):
            print(f"{image_folder_path} is exit")
            continue
                
        image_files = os.listdir(folder_path+"/"+image_folder_path)
        with open('out_print.txt', 'a') as f:
            print(str(j) + ": " + image_folder_path, file=f)
        with open(folder_path + "/" + image_folder_path + '/_caption.txt', 'w') as f:
            for k in range(len(image_files)):
                # JPEG画像を開く
                image_path = image_files[k]
                if image_path[-1] == "t":
                    continue
                raw_image = Image.open(folder_path + "/" + image_folder_path + "/" + image_path)
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                generated_text = lavis_model.generate({"image": image}, use_nucleus_sampling=True,num_captions=1)[0]
                print(image_path[:-4], file=f)
                print(generated_text, file=f)


def test():
    A = torch.zeros(11, 3, 2, 2)
    for i in range(11):
        for j in range(3):
            for k in range(2):
                for l in range(2):
                    A[i][j][k][l] = i*11+j
          
    print(A.shape)          
    print(A)
    
    B = torch.cat([A[i] for i in range(11)], dim=0)
    print(B.shape)
    print(B)
        
    
    
if __name__ == '__main__':
    data_path = "PictureSelector/data" 
    print(data_path)
    #captioning(data_path)
    
    n = 10  # Number of photos (n+1) for each input
    photo_channels = 3  # Number of channels for each photo (e.g., RGB)
    photo_size = 256  # Size of each photo (assuming square photos)
    n_photos = n + 1
    
    #test()

    # Train and save the model
    trained_model_path = 'trained_model'
    model = train_cnn_photo_removal_network(n_photos, n_epochs=1000000, save_path=trained_model_path)
    #model = train_cnn_photo_removal_network(n_photos, save_path=trained_model_path)

    # Load the saved model
    #loaded_model = load_cnn_photo_removal_network(trained_model_path)
    
    print("########## FINISH !!!!! ############")