import pandas as pd
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as Meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords

from habitat.core.logging import logger

# 必要なNLTKのリソースをダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


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
        #logger.info(f"sentence_list={len(sentence_list)}")
        # 文章をSBERTで埋め込みベクトルに変換
        embeddings = [self.sbert.encode(sentence, convert_to_tensor=True).unsqueeze(0) for sentence in sentence_list]
        # embeddingsの形状を確認
        #logger.info(f"embeddings_shape={[embed.shape for embed in embeddings]}")
    

       # 6つのベクトルを結合 (次元を6倍にする)
        combined_features = torch.cat(embeddings, dim=1)
        
        # MLPを通してスカラー値を予測
        output = self.mlp(combined_features)
        return output

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

def get_human_eval_df(human_eval_path):
    human_eval_df = pd.read_csv(human_eval_path, header=None, names=["scene_id", "score", "desc_name", "sentence"])
    return human_eval_df

# 学習済みモデルから評価を算出する(一気に5つのGT-descriptionの評価を出す)
def calculate_train_5score(description_dict, human_eval_df, mode, shuffle=False, ckpt=10000):
    logger.info("Start calculating Human-Enhanced Similarity 5 Score...")
    
    # GPUが使える場合はGPUに、それ以外の場合はCPUにデータを転送
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    header = [
        "scene_id", "desc_name", "score", "hes_score"
    ]

    add_word = ""
    if shuffle == True:
        add_word = "_shuffle"

    model_path = f"./../SentenceBert_FineTuning/model_checkpoints{add_word}/model_epoch_{ckpt}.pth"
    if mode == "":
        model_path = f"./../SentenceBert_FineTuning/model_checkpoints{add_word}/model_epoch_{ckpt}.pth"
    elif mode == "2":
        model_path = f"./../SentenceBert_FineTuning/model_checkpoints_all{add_word}/model_epoch_{ckpt}.pth" 
    
    # SBERTモデルのロード
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SBERTRegressionModel(sbert_model).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    
    with open(f"./result_score_hes{mode}{add_word}.csv", mode="w") as f:
        f.write(",".join(header) + "\n")
        for _, row in human_eval_df.iterrows():
            scene_id = row["scene_id"]
            sentence = row["sentence"]
            logger.info(f"scene: {scene_id}")
            
            sentence_list = [sentence]
            for i in range(5):
                gt_description = description_dict[scene_id][i]
                sentence_list.append(gt_description)
                
            hes_score = model(sentence_list).item()
                
            out_row = [
                scene_id, row['desc_name'], str(row['score']), f"{hes_score:.5f}",
            ]
            f.write(",".join(out_row) + "\n")
            logger.info(out_row)

# 学習済みモデルから評価を算出する(1つのGT-descriptionから評価を計算して平均する)
def calculate_train_eachscore(description_dict, human_eval_df, mode, ckpt=10000):
    pass

if __name__ == "__main__":
    sentence_path = "./../data/scene_datasets/mp3d/Environment_Descriptions.txt"
    human_eval_path = "./description_average.csv"
    
    description_dict = get_description_dict(sentence_path)
    human_eval_df = get_human_eval_df(human_eval_path)
    
    #calculate_metrics(description_dict, human_eval_df)
    #calculate_nominal_semantic_score(description_dict, human_eval_df)
    #calculate_nominal_adjective_semantic_score(description_dict, human_eval_df)
    #calculate_scene_object_matching_score(human_eval_df)
    #calculate_weighted_f_score(description_dict, human_eval_df)

    mode = ""
    mode = "2"
    shuffle = True
    calculate_train_5score(description_dict, human_eval_df, mode, shuffle)

    logger.info("Finish !!")
