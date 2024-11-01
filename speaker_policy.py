import os
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from habitat.core.logging import logger

from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess
import clip

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as Meteor_score
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


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
        embeddings = [self.sbert.encode(sentence, convert_to_tensor=True).unsqueeze(0) for sentence in sentence_list]

        # 6つのベクトルを結合 (次元を6倍にする)
        combined_features = torch.cat(embeddings, dim=1)
        
        # MLPを通してスカラー値を予測
        output = self.mlp(combined_features)
        return output


device = torch.device('cuda')
# lavisモデルの読み込み
lavis_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
lavis_model.to(device)

# Sentence-BERTモデルの読み込み
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_model.to(device)

# Load the clip model
clip_model, preprocess = clip.load('ViT-B/32', device)

# SBERTモデルのロード
model_path = f"/gs/fs/tga-aklab/matsumoto/Main/SentenceBert_FineTuning/model_checkpoints_all/model_epoch_10000.pth"
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
eval_model = SBERTRegressionModel(sbert_model).to(device)
eval_model.load_state_dict(torch.load(model_path))
eval_model.eval() 
logger.info(f"Eval Model loaded from {model_path}")

# 単語のステミング処理
lemmatizer = WordNetLemmatizer()

def create_description_dict():
    # ファイルを読み込んで行ごとにリストに格納する
    with open('data/scene_datasets/mp3d/Environment_Descriptions.txt', 'r') as file:
        lines = file.readlines()

        # scene id と文章を抽出してデータフレームに変換する
        description_dict = {}
        for i in range(0, len(lines), 7):
            descriptions = []
            scene_id = lines[i].strip()
            desc_ind = i+2
            for j in range(5):
                descriptions.append(lines[desc_ind+j].strip())
            description_dict[scene_id] = descriptions

    return description_dict

def create_caption(picture):
    # 画像からcaptionを生成する
    image = Image.fromarray(picture)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    generated_text = lavis_model.generate({"image": image}, use_nucleus_sampling=True,num_captions=1)[0]
    return generated_text

def calculate_similarity(pred_description, origin_description):
    # 文をSentence Embeddingに変換
    embedding1 = bert_model.encode(pred_description, convert_to_tensor=True)
    embedding2 = bert_model.encode(origin_description, convert_to_tensor=True)
    
    # コサイン類似度を計算
    sentence_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return sentence_sim

def create_new_image_embedding(obs):
    image = Image.fromarray(obs)
    image = preprocess(image)
    image = torch.tensor(image).clone().detach().to(device).unsqueeze(0)
    embetting = clip_model.encode_image(image).float()
    return embetting

def decide_save(emd, results):
    select_threthould = 0.9
    #select_threthould = 0.8
    for i in range(len(results)):
        check_emb = create_new_image_embedding(results[i][1])

        sim = util.pytorch_cos_sim(emd, check_emb).item()
        if sim >= select_threthould:
            return False
    return True

def select_pictures(picture_list):
    # 提案手法
    # picture_list[0]: picture_value
    # picture_list[1]: rgb
    results = []
    num_picture = 10
    num_picture = 100
    logger.info(f"num_picture = {num_picture}")

    sorted_picture_list = sorted(picture_list, key=lambda x: x[0], reverse=True)
    i = 0
    while True:
        if len(results) == num_picture:
            break
        if i == len(sorted_picture_list):
            break
        emd = create_new_image_embedding(sorted_picture_list[i][1])
        is_save = decide_save(emd, results)

        if is_save == True:
            results.append(sorted_picture_list[i])
        i += 1

    return results

def select_by_depth(picture_list, depth_threshold):
    results = []
    
    for pic_list in picture_list:
        depth_value = pic_list[2]
        if depth_value >= depth_threshold:
            results.append(pic_list)

    return results

def select_by_object(picture_list, object_threshold):
    results = []
    
    for pic_list in picture_list:
        object_num = pic_list[3]
        if object_num >= object_threshold:
            results.append(pic_list)

    return results

def select_by_activation(picture_list, activation_threshold, i):
    logger.info("Activation")
    results = []
    #logger.info(clip_model)
    data = []
    
    for pic_list in picture_list:
        image = pic_list[1]
        image = Image.fromarray(image)

        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        #logger.info(f"lavis_model.visual_encoder={lavis_model.visual_encoder}")

        # 特徴マップの取得(blip_2)
        with torch.no_grad():
            # モデルの視覚エンコーダーに画像を入力
            #visual_features = lavis_model.visual_encoder(image)
            x = lavis_model.visual_encoder

            B = image.shape[0]
            x = lavis_model.visual_encoder.patch_embed(image)

            cls_tokens = lavis_model.visual_encoder.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            x = x + lavis_model.visual_encoder.pos_embed[:, : x.size(1), :]
            x = lavis_model.visual_encoder.pos_drop(x)

            for j, blk in enumerate(lavis_model.visual_encoder.blocks):
                x = blk(x, -1 == j)

            image_features = x

        """
        # 特徴マップの取得(blip_3)
        with torch.no_grad():
            #logger.info(f"image={image.shape}")
            image_features = lavis_model.forward_encoder({"image": image})
        """
        #logger.info(f"image_features={image_features.shape}")
        mean = image_features.mean().item()
        if mean >= activation_threshold:
            results.append(pic_list)

        #data.append(mean)

    #create_histogram(data, i)
    #create_histogram(data, i, filename="histogram_blip2")

    return results

def create_histogram(data, i, filename='histogram'):
    # ヒストグラムを作成
    plt.hist(data, bins=10, alpha=0.7, color='blue', edgecolor='black')
    
    # タイトルとラベルを設定
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # 画像を保存
    plt.savefig(f"{filename}_{i}.png")
    plt.close()
    print(f'ヒストグラムを {filename} として保存しました。')

def load_images_and_extract_metadata(dir_name):
    obs_list = []

    # ファイル名のパターンを定義
    pattern = re.compile(r"(?P<scene_name>.+?)_(?P<j>\d+)_(?P<depth>[\d.]+)_(?P<obs_num>[\d.]+)_(?P<pic_value>[\d.]+)_(?P<area>[\d.]+)\.png")

    # ディレクトリ内のファイルをループ
    for filename in os.listdir(dir_name):
        match = pattern.match(filename)
        if match:
            # ファイルのメタデータを抽出し、floatに変換
            scene_name = match.group("scene_name")
            depth = float(match.group("depth"))
            obs_num = float(match.group("obs_num"))
            pic_value = float(match.group("pic_value"))
            area = float(match.group("area"))

            # 画像を読み込んでNumPy配列に変換
            img_path = os.path.join(dir_name, filename)
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)

            obs = [pic_value, img, depth, obs_num, area]
            obs_list.append(obs)

    return obs_list, scene_name

def calculate_pas(s_lemmatized, description):
    gt_lemmatized = lemmatize_and_filter(description)
    precision, recall, total_weight, total_gt_weight = 0.0, 0.0, 0.0, 0.0
    matched_words = set()

    for j, s_word in enumerate(s_lemmatized):
        weight = 1.0 / (j + 1)  # 単語の位置に応じた重み付け
        total_weight += weight
                
        if any(is_matching(s_word, gt_word) for gt_word in gt_lemmatized):
            precision += weight
            matched_words.add(s_word)

    for j, gt_word in enumerate(gt_lemmatized):
        weight = 1.0 / (j + 1)
        total_gt_weight += weight
        if any(is_matching(gt_word, s_word) for s_word in matched_words):
            recall += weight

    precision /= total_weight if total_weight > 0 else 1
    recall /= total_gt_weight if total_gt_weight > 0 else 1

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    return f_score

# 単語が一致しているかどうかを判断する
def is_matching(word1, word2):
    # ステミングされた単語が一致するか
    lemma1 = lemmatizer.lemmatize(word1)
    lemma2 = lemmatizer.lemmatize(word2)
        
    if lemma1 == lemma2:
        return True
        
    # 類義語が存在するか
    synsets1 = wordnet.synsets(lemma1)
    synsets2 = wordnet.synsets(lemma2)
        
    if synsets1 and synsets2:
        # synsetsをリーマティックに比較
        return any(s1.wup_similarity(s2) >= 0.9 for s1 in synsets1 for s2 in synsets2)
        
    return False

def get_wordnet_pos(word):
    """WordNetの品詞タグを取得"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_and_filter(text):
    """ステミング処理を行い、ストップワードを除去"""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
                    for token in tokens if token.isalpha() 
                    and token not in stopwords.words('english')]
    return filtered_tokens

def get_txt2dict(txt_path):
    data_dict = {}
    # ファイルを読み込み、行ごとにリストに格納
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # 奇数行目をキー、偶数行目を値として辞書に格納
    for i in range(0, len(lines), 2):
        scene_name = lines[i].strip()  # 奇数行目: scene名
        scene_data = lines[i + 1].strip().split(',')  # 偶数行目: コンマ区切りのデータ
        data_dict[scene_name] = scene_data
        
    return data_dict

# 名詞であるかどうかを判断するための追加のフィルター
def is_valid_noun(word):
    """単語が名詞であるかを確認する追加のフィルター"""
    # 除外したい名詞のリスト
    excluded_nouns = {"inside", "lead", "use", "look", "like", "lot", "clean", "middle", "walk", "gray"}

    if word in excluded_nouns:
        return False
    synsets = wordnet.synsets(word)
    return any(s.pos() == 'n' for s in synsets)

# sentence内の名詞のリストを取得
def extract_nouns(sentence):
    tokens = word_tokenize(sentence)
    nouns = []

    for word in tokens:
        if word.isalpha() and word not in stopwords.words('english'):
            # 原型に変換
            lemma = lemmatizer.lemmatize(word)
            pos = get_wordnet_pos(word)
            if pos == wordnet.NOUN and is_valid_noun(lemma):  # 名詞に限定
                if lemma not in nouns:
                    nouns.append(lemma)

    return nouns

def calculate_clip_score(image, text):
    # 画像の読み込み
    image = Image.fromarray(image)
    
    # 画像の前処理
    inputs = preprocess(image).unsqueeze(0).to(device)

    # テキストのトークン化とエンコード
    text_tokens = clip.tokenize([text]).to(device)

    # 画像とテキストの特徴ベクトルを計算
    with torch.no_grad():
        image_features = clip_model.encode_image(inputs)
        text_features = clip_model.encode_text(text_tokens)

    # 類似度（cosine similarity）を計算
    clip_score = torch.cosine_similarity(image_features, text_features)
    
    return clip_score.item()

def calculate_iou(word1, word2):
    # word1, word2 の同義語集合を取得し、それらのJaccard係数を用いてIoU計算を行います。
    synsets1 = set(wordnet.synsets(word1))
    synsets2 = set(wordnet.synsets(word2))
    intersection = synsets1.intersection(synsets2)
    union = synsets1.union(synsets2)
    if not union:  # 同義語が全くない場合は0を返す
        return 0.0
    return len(intersection) / len(union)

# IoU行列の生成
def generate_iou_matrix(object_list1, object_list2):
    iou_matrix = np.zeros((len(object_list1), len(object_list2)))
    for i, obj1 in enumerate(object_list1):
        for j, obj2 in enumerate(object_list2):
            iou_matrix[i, j] = calculate_iou(obj1, obj2)
    return iou_matrix

# Jonker-Volgenantアルゴリズム（線形代入問題の解法）で最適な対応を見つける
def find_optimal_assignment(object_list1, object_list2):
    iou_matrix = generate_iou_matrix(object_list1, object_list2)
    # コスト行列はIoUの負の値を使う（最小コストの最大化）
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_iou = iou_matrix[row_ind, col_ind].sum() / min(len(object_list1), len(object_list2))
    return optimal_iou, list(zip(row_ind, col_ind))

def calculate_ed(object_list, pred_sentence, area, clip_score):
    pred_object_list = extract_nouns(pred_sentence)

    if len(pred_object_list) == 0:
        logger.info(f"len(pred_object_list)=0")
        return 0.0
    
    optimal_iou, assignment = find_optimal_assignment(object_list, pred_object_list)

    #logger.info(f"Object_list: {object_list}")
    #logger.info(f"Pred_list: {pred_object_list}")
    #logger.info(f"Assignment: {assignment}")

    ed_score = clip_score * optimal_iou * area
    logger.info(f"ED-S: {ed_score}, CLIP Score: {clip_score}, IoU: {optimal_iou}, Area: {area}")

    return ed_score


if __name__ == "__main__":
    description_dict = create_description_dict()
    scene_object_dict = get_txt2dict("/gs/fs/tga-aklab/matsumoto/Main/scene_object_list.txt")
    selection_method = "Proposed"
    selection_method = "Depth"
    #selection_method = "Object"
    #selection_method = "Always"
    #selection_method = "Activation"
    logger.info(f"Selection: {selection_method}")

    similarity_list = []
    hes_list = []
    pas_list = []
    ed_list = []
    loq_list = []
    for i in range(85, 111):
        dir_name = f"/gs/fs/tga-aklab/matsumoto/Main/collected_images/{i}/"
        picture_list, scene_name = load_images_and_extract_metadata(dir_name)
        #logger.info(f"picture_list = {len(picture_list)}")
        logger.info(f"{i}: SCENE: {scene_name}")
        gt_list = description_dict[scene_name]
        object_list = scene_object_dict[scene_name]
        
        selected_list = []
        if selection_method == "Proposed":
            selected_list = select_pictures(picture_list)
            logger.info(f"selected_list = {len(selected_list)}")
        elif selection_method == "Depth":
            depth_threshold = 3.5
            logger.info(f"Depth >= {depth_threshold}")
            selected_list = select_by_depth(picture_list, depth_threshold)
            logger.info(f"selected_list = {len(selected_list)}")
        elif selection_method == "Object":
            object_threshold = 6.0
            logger.info(f"Object >= {object_threshold}")
            selected_list = select_by_object(picture_list, object_threshold)
            logger.info(f"selected_list = {len(selected_list)}")
        elif selection_method == "Always":
            logger.info(f"Always")
            selected_list = picture_list
            logger.info(f"selected_list = {len(selected_list)}")
        elif selection_method == "Activation":
            Activation_threshold = 0.0
            logger.info(f"Activation >= {Activation_threshold}")
            selected_list = select_by_activation(picture_list, Activation_threshold, i)
            logger.info(f"selected_list = {len(selected_list)}")

        loq_list.append(len(selected_list))
        #logger.info(f"selected_list = {len(selected_list)}")
        #logger.info(f"{len(selected_list[0])}")

        skip_flag = False
        if len(selected_list) == 0:
            skip_flag = True

        pred_sentence = ""
        hes_sentence_list = []
        clip_score_list = []
        area = 0.0
        
        for j in range(len(selected_list)):
            pic_list = selected_list[j]
            caption = create_caption(pic_list[1])
            clip_score_list.append(calculate_clip_score(pic_list[1], caption))
            pred_sentence += (caption + ". ")
            area = pic_list[4]
        
        
        hes_sentence_list.append(pred_sentence)
        s_lemmatized = lemmatize_and_filter(pred_sentence) 
        if len(clip_score_list) == 0:
            clip_score = 0
        else:
            clip_score = sum(clip_score_list) / len(clip_score_list)                      

        for j in range(len(gt_list)):
            gt_sentence = gt_list[j]
            hes_sentence_list.append(gt_sentence)

            if skip_flag == True:
                similarity_list.append(0)
                pas_list.append(0)
                continue
                
            sim = calculate_similarity(pred_sentence, gt_sentence)
            similarity_list.append(sim)

            pas = calculate_pas(s_lemmatized, gt_sentence)
            pas_list.append(pas)

        hes = eval_model(hes_sentence_list).item()
        hes_list.append(hes)
        ed = calculate_ed(object_list, pred_sentence, area, clip_score)
        ed_list.append(ed)

        logger.info(f"Sim: {sum(similarity_list)}, PAS: {sum(pas_list)}, HES: {sum(hes_list)}, ED-S: {sum(ed_list)}, LOQ: {sum(loq_list)}")
        logger.info(f"{len(similarity_list)}, {len(pas_list)}, {len(hes_list)}, {len(ed_list)}, {len(loq_list)}")

    similarity_score = sum(similarity_list) / len(similarity_list)
    pas_score = sum(pas_list) / len(pas_list)
    hes_score = sum(hes_list) / len(hes_list)
    ed_score = sum(ed_list) / len(ed_list)
    loq_score = sum(loq_list) / len(loq_list)

    logger.info(f"Similarity: {similarity_score}")
    logger.info(f"PAS Score: {pas_score}")
    logger.info(f"HES Score: {hes_score}")
    logger.info(f"ED-S: {ed_score}")
    logger.info(f"LOQ: {loq_score}")
        

            



        

