import os
import datetime
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from log_manager import LogManager
from habitat.core.logging import logger

# SBERT + MLPによる回帰モデルの定義
class SBERTRegressionModel(nn.Module):
    def __init__(self, sbert_model, hidden_size1=512, hidden_size2=256, hidden_size3=128):
        super(SBERTRegressionModel, self).__init__()
        self.sbert = sbert_model
        
        # 6つの埋め込みベクトルを結合するため、入力サイズは6倍に
        embedding_size = self.sbert.get_sentence_embedding_dimension() * 6
        
        # 多層MLPの構造を定義
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size1),  # 結合ベクトルから第1隠れ層
            nn.ReLU(),  # 活性化関数
            nn.Linear(hidden_size1, hidden_size2),  # 第2隠れ層
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),  # 第3隠れ層
            nn.ReLU(),
            nn.Linear(hidden_size3, 1)  # 隠れ層からスカラー値出力
        )
        
    def forward(self, sentence_list):
        # 文章をSBERTで埋め込みベクトルに変換
        embeddings = [self.sbert.encode(sentence, convert_to_tensor=True) for sentence in sentence_list]
        
       # 6つのベクトルを結合 (次元を6倍にする)
        combined_features = torch.cat(embeddings, dim=1)
        
        # MLPを通してスカラー値を予測
        output = self.mlp(combined_features)
        return output
    
class TextGroupDataset(Dataset):
    def __init__(self, sentence_groups, labels):
        self.sentence_groups = sentence_groups
        self.labels = labels

    def __len__(self):
        return len(self.sentence_groups)

    def __getitem__(self, idx):
        sentence_list = self.sentence_groups[idx]
        label = self.labels[idx]
        return sentence_list, label
    
# 保存したモデルのロード方法
def load_model(epoch_num, model, checkpoint_dir="model_checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Model loaded from {checkpoint_path}")
    return model

def get_description_dict(sentence_path):
    # ファイルを読み込んで行ごとにリストに格納する
    with open(sentence_path, 'r') as file:
        lines = file.readlines()

    # scene idと文章を抽出してdictに格納する
    description_dict = {}
    for i in range(0, len(lines), 7):
        descriptions = []
        scene_id = lines[i].strip()
        desc_ind = i+2
        for j in range(5):
            descriptions.append(lines[desc_ind+j].strip())
            
        description_dict[scene_id] = descriptions

    return description_dict

def create_sentence_groups():
    sentence_groups_train = []
    sentence_groups_test = []
    labels_train = []
    labels_test = []
    gt_description_path = "./../data/scene_datasets/mp3d/Environment_Descriptions.txt"
    human_result_path = "./human_results.csv"
    test_scene = ["Uxmj2M2itWa", "V2XKFyX4ASd", "VFuaQ6m2Qom"]
    
    description_dict = get_description_dict(gt_description_path)
    header = ["scene_name", "description_index", "line", "score", "description_name", "description"]
    human_df = pd.read_csv(human_result_path, header=0, names=header)
    #print(human_df.head())
    
    for _, row in human_df.iterrows():
        sentences = []
        description = row["description"]
        scene_name = row["scene_name"]
        gt_description_list = description_dict[scene_name]
        
        sentences.append(description)
        for gt in gt_description_list:
            sentences.append(gt)
        
        if scene_name in test_scene:
            sentence_groups_test.append(sentences)
            labels_test.append(row["score"])    
        else:
            sentence_groups_train.append(sentences)
            labels_train.append(row["score"])
        
    labels_train = torch.tensor(labels_train, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    return sentence_groups_train, sentence_groups_test, labels_train, labels_test

if __name__ == "__main__":
    # Sentence Bertのパラメータは更新しない
    logger.info("FineTuning 1")
    start_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    logger.info(f"Start at {start_date}")
    
    
    log_manager = LogManager()
    train_logger = log_manager.createLogWriter("train")
    test_logger = log_manager.createLogWriter("test")

    # GPUが使える場合はGPUに、それ以外の場合はCPUにデータを転送
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    # SBERTモデルのロード
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # SBERTのパラメータをフリーズ（更新しない）
    for param in sbert_model.parameters():
        param.requires_grad = False
        
    # SBERT + MLPの回帰モデルを初期化
    model = SBERTRegressionModel(sbert_model).to(device)

    sentence_groups_train, sentence_groups_test, labels_train, labels_test = create_sentence_groups()

    """
    # データセットを学習用と評価用に分割
    sentence_groups_train, sentence_groups_test, labels_train, labels_test = train_test_split(
        sentence_groups, labels, test_size=0.2, random_state=42)
    """

    # データセットとデータローダーを作成
    train_dataset = TextGroupDataset(sentence_groups_train, labels_train)
    test_dataset = TextGroupDataset(sentence_groups_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # OptimizerとLossの設定
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss().to(device)
    
    # モデルのパラメータを保存するディレクトリを作成
    save_dir = "model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # トレーニングとモデル保存
    epochs = 10000
    logger.info(f"Start epoch={epochs}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for sentence_list, labels in train_loader:  # sentence_listは6つの文章のリスト
            #sentence_list = [sentence[0].to(device) for sentence in sentence_list]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sentence_list)  # 6つの文章を入力
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 10エポックごとにモデルを保存
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')
            train_logger.writeLine(f"{epoch+1},{total_loss/len(train_loader)}")

        if (epoch + 1) % 100 == 0:
            # モデルの評価
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for sentence_list, labels in test_loader:
                    #sentence_list = [sentence[0].to(device) for sentence in sentence_list]
                    labels = labels.to(device)

                    outputs = model(sentence_list)
                    loss = criterion(outputs.squeeze(), labels)
                    total_test_loss += loss.item()

            logger.info(f"Test Loss at epoch {epoch+1}: {total_test_loss/len(test_loader)}")
            test_logger.writeLine(f"{epoch+1},{total_test_loss/len(test_loader)}")
            
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved at {checkpoint_path}")

    # 評価データの損失を表示
    logger.info(f"Final Test Loss: {total_test_loss/len(test_loader)}")
    end_date = datetime.datetime.now().strftime('%y-%m-%d %H-%M-%S') 
    logger.info(f"End at {end_date}")