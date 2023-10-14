import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib

img1 = cv2.imread('result/val/heat_map/23-08-08 21-46-33.png')
img2 = cv2.imread('figures/cal_CI/fig_rgb3.png')
print(img1.shape)
print(img2.shape)
assert img1.shape == img2.shape, "2つの画像のサイズは一致していなければならない"

# アルファブレンディング
alpha = 0.45
blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

# 結果を表示する。
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis('off')
#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/val/heat_map")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/val/heat_map/heat_picture.png')

#グラフの表示
plt.show()