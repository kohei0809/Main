import os
import torch
import cv2
import numpy as np
from torchvision import transforms, utils, models
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats

from TranSalNet.utils.data_process import preprocess_img, postprocess_img
from TranSalNet.TranSalNet_Dense import TranSalNet


def create_saliency_map(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TranSalNet()
    model.load_state_dict(torch.load('TranSalNet/pretrained_models/TranSalNet_Dense.pth'))

    model = model.to(device) 
    model.eval()

    img = preprocess_img(img_dir=image_path) # padding and resizing input image into 384x288
    img = np.array(img)/255.
    img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
    img = torch.from_numpy(img)
    if torch.cuda.is_available():
        img = img.type(torch.cuda.FloatTensor).to(device)
    else:
        img = img.type(torch.FloatTensor).to(device)
    pred_saliency = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    pred_saliency = postprocess_img(pic, image_path) # restore the image to its original size as the result
    high_saliency = pred_saliency[pred_saliency >= 128]
    # 0を削除
    non_zero_pred_saliency = pred_saliency[pred_saliency != 0]
    flag = (stats.mode(non_zero_pred_saliency).mode == 1)
    print(f"image_path: {image_path}, {flag}, {int(pred_saliency.mean())}, {pred_saliency.max()}, {int(high_saliency.shape[0]/(pred_saliency.shape[0]*pred_saliency.shape[1])*1000)}")

    img_name = os.path.basename(image_path)
    os.makedirs("TranSalNet/saliency_maps", exist_ok=True)
    cv2.imwrite("TranSalNet/saliency_maps/" + img_name, pred_saliency, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) # save the result

    # 元画像の上にSaliency Mapを重ねる
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap = cv2.applyColorMap(pred_saliency, cv2.COLORMAP_JET)
    overlay_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    os.makedirs("TranSalNet/overlayed_images", exist_ok=True)
    cv2.imwrite("TranSalNet/overlayed_images/" + img_name, overlay_image)
    
    # ヒストグラムを計算
    histogram, bins = np.histogram(non_zero_pred_saliency.flatten(), bins=range(257))  # ピクセル値ごとの出現回数
    #histogram, bins = np.histogram(pred_saliency.flatten(), bins=range(257))

    # 棒グラフを描画
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], histogram, width=1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Histogram')
    plt.grid(True)
    os.makedirs("TranSalNet/saliency_histgram", exist_ok=True)
    plt.savefig(f"TranSalNet/saliency_histgram/{img_name}_.png")
    plt.clf()
    

if __name__ == "__main__":
    image_folder = "TranSalNet/pictures"
    for image_name in os.listdir(image_folder):
        if image_name[-1] == "g":
            image_path = os.path.join(image_folder, image_name)
            create_saliency_map(image_path)
    print("Finished")