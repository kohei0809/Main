import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as Meteor_score
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from habitat.core.logging import logger

# 必要なNLTKのリソースをダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# 単語のステミング処理
lemmatizer = WordNetLemmatizer()

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

def sentence_embedding_similarity(sentence1, sentence2):
    # Sentence-BERTモデルの読み込み
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 文をSentence Embeddingに変換
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # コサイン類似度を計算
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return similarity

# BLEUスコアの計算
def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

# ROUGEスコアの計算
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# METEORスコアの計算
def calculate_meteor(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    return Meteor_score([reference], candidate)

def lemmatize_and_filter(text):
    """ステミング処理を行い、ストップワードを除去"""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
                    for token in tokens if token.isalpha() 
                    and token not in stopwords.words('english')]
    return filtered_tokens

def get_wordnet_pos(word):
    """WordNetの品詞タグを取得"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

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

# PAS_Scoreの計算
def calculate_pas(description, pred_description):
    s_lemmatized = lemmatize_and_filter(pred_description)                        
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

def read_file_to_list(file_path):
    input_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 改行文字を取り除いてリストに追加
            input_list.append(line.strip())
    return input_list


# テスト用の文
gt_sentence_list = read_file_to_list("sentences/gt_sentences.txt")
pred_sentence_list = []
for i in range(11):
    pred_sentence_list.append(read_file_to_list(f"sentences/pred_sentence_{i}.txt"))
name_list = ["matsumoto", "kondo", "nakamura", "aizawa", "edward"]

device = (
    torch.device("cuda", 0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
model_path = f"/gs/fs/tga-aklab/matsumoto/Main/SentenceBert_FineTuning/model_checkpoints_all/model_epoch_10000.pth"
# SBERTモデルのロード
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
eval_model = SBERTRegressionModel(sbert_model).to(device)
eval_model.load_state_dict(torch.load(model_path))
eval_model.eval() 

for scene_idx in range(50, 55, 5):
    for idx, name in enumerate(name_list):
        logger.info(f"-------------- {name}_{int(scene_idx/5)} --------------")
        pred_sentence = pred_sentence_list[int(scene_idx/5)][idx]
        logger.info(f"pred: {pred_sentence}")
        similarity_list = []
        bleu_list = []
        rouge_1_list = []
        rouge_2_list = []
        rouge_L_list = []
        meteor_list = []
        pas_list = []
        hes_sentence_list = [pred_sentence]
        
        for desc_num in range(5):
            gt_sentence = gt_sentence_list[scene_idx+desc_num]
            logger.info(f"GT: {gt_sentence}")
            hes_sentence_list.append(gt_sentence)
          
            # Sentence-BERTを利用した類似度を計算
            similarity = sentence_embedding_similarity(gt_sentence, pred_sentence)
            similarity_list.append(similarity)
            
            # BLUEスコアを計算
            bleu_score = calculate_bleu(gt_sentence, pred_sentence)
            bleu_list.append(bleu_score)
            
            # ROUGEスコアを計算
            rouge_scores = calculate_rouge(gt_sentence, pred_sentence)
            rouge_1_score = rouge_scores['rouge1'].fmeasure
            rouge_2_score = rouge_scores['rouge2'].fmeasure
            rouge_L_score = rouge_scores['rougeL'].fmeasure
            rouge_1_list.append(rouge_1_score)
            rouge_2_list.append(rouge_2_score)
            rouge_L_list.append(rouge_L_score)
            
            # METEORスコアを計算
            meteor_score = calculate_meteor(gt_sentence, pred_sentence)
            meteor_list.append(meteor_score)
            
            # PAS_Scoreを計算
            pas_score = calculate_pas(gt_sentence, pred_sentence)
            pas_list.append(pas_score)
            
        similarity_score = sum(similarity_list) / len(similarity_list)
        bleu_score = sum(bleu_list) / len(bleu_list)
        rouge_1_score = sum(rouge_1_list) / len(rouge_1_list)
        rouge_2_score = sum(rouge_2_list) / len(rouge_2_list)
        rouge_L_score = sum(rouge_L_list) / len(rouge_L_list)
        meteor_score = sum(meteor_list) / len(meteor_list)
        pas_score = sum(pas_list) / len(pas_list)
        hes_score = eval_model(hes_sentence_list).item()
        
        logger.info(f"-------------- {name}_{int(scene_idx/5)} --------------")
        logger.info(f"Sentence Embedding Similarity between gt_sentence and pred_sentence: {similarity_score:.5f}")
        logger.info(f"BLEU score between gt_sentence and pred_sentence: {bleu_score:.5f}")
        logger.info(f"ROUGE score between gt_sentence and pred_sentence: rouge1= {rouge_1_score:.5f} rouge2= {rouge_2_score:.5f} rougeL= {rouge_L_score:.5f}")
        logger.info(f"METEOR score between gt_sentence and pred_sentence: {meteor_score:.5f}")
        logger.info(f"PAS score between gt_sentence and pred_sentence: {pas_score:.5f}")
        logger.info(f"HES score between gt_sentence and pred_sentence: {hes_score:.5f}")
