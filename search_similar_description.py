import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as Meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import itertools
from collections import Counter
import datetime

from habitat.core.logging import logger

def sentence_embedding_similarity(sentence1, sentence2):
    # Sentence-BERTモデルの読み込み
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 文をSentence Embeddingに変換
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # コサイン類似度を計算
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

def search_similar_description(df, search_keyword):
    logger.info("Start search...")
    max_similarity = -1
    second_similarity = 0
    third_similarity = 0
    max_idx = -1
    second_idx = -1
    third_idx = -1
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"idx: {idx}")

        scene_id = row["scene_id"]
        sentence = row["sentence"]
        #print(f"scene: {scene_id}")
                
        # Similarityを計算
        similarity = sentence_embedding_similarity(search_keyword, sentence)
        
        if similarity > max_similarity:
            third_similarity = second_similarity
            second_similarity = max_similarity
            max_similarity = similarity
            third_idx = second_idx
            second_idx = max_idx
            max_idx = idx
    
    logger.info(f"Max Similarity: {max_similarity:.5f}")
    logger.info(f"scene_id: {df.iloc[max_idx]['scene_id']}")
    logger.info(f"sentence: {df.iloc[max_idx]['sentence']}")  
    logger.info(f"Second Similarity: {second_similarity:.5f}")
    logger.info(f"scene_id: {df.iloc[second_idx]['scene_id']}")
    logger.info(f"sentence: {df.iloc[second_idx]['sentence']}")  
    logger.info(f"Third Similarity: {third_similarity:.5f}")
    logger.info(f"scene_id: {df.iloc[third_idx]['scene_id']}")
    logger.info(f"sentence: {df.iloc[third_idx]['sentence']}")      
            
def input_df(date):
    file_path = f"log/{date}/eval/description.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    lines = [line.strip() for line in lines]

    # 5行ごとにデータを分割
    data = {'scene_id': [], 'sentence': []}
    for i in range(0, len(lines), 5):
        #print(f"i={i}")
        scene_name = lines[i][:11]
        data['scene_id'].append(scene_name)
        #data['gt_description'].append(lines[i+1])
        data['sentence'].append(lines[i+2])
        #data['value'].append(lines[i+3])
        
    # DataFrameに変換
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    #date = "24-10-25 10-54-43"
    date = "24-11-06 03-47-13"
    date = "24-11-10 01-04-16"
    #search_keyword = "A house with good sunlight and a large space for gatherings"
    search_keyword = "A quiet house that is perfect for remote work"
    #search_keyword = "A house with plenty of sunlight and comfortable even for large groups"
    #search_keyword = "A building with a good view that can accommodate large groups of people"
    #search_keyword = "A house with a good view where even a large family can live comfortably"
    #search_keyword = "A place where you can work without any problems and do creative work"
    
    now = datetime.datetime.now()
    logger.info(f"date={date}")
    logger.info(now)
    logger.info(f"keyword: {search_keyword}")
    
    df = input_df(date)
    
    search_similar_description(df, search_keyword)
    now = datetime.datetime.now()
    logger.info(now)
