import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date1 = "24-03-08 14-49-01"
date2 = "24-03-13 01-04-07"
date3 = "24-03-16 20-40-27"
date4 = "24-03-09 01-05-42"

df1 = pd.read_csv("log/" + date1 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'each_sim', 'path_length'], header=None)
plt.plot(df1['time'], df1['similarity'], color="blue", label="Dense reward")
df2 = pd.read_csv("log/" + date2 + "/train2/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'each_sim', 'path_length'], header=None)
plt.plot(df2['time'], df2['similarity'], color="red", label="Sparse reward")
df3 = pd.read_csv("log/" + date3 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'each_sim', 'path_length'], header=None)
plt.plot(df3['time'], df3['similarity'], color="green", label="Dense reward (Random Selection)")
df4 = pd.read_csv("log/" + date4 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'each_sim', 'path_length'], header=None)
plt.plot(df4['time'], df4['similarity'], color="black", label="Dense reward ($N_p=5$)")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Sentence Similarity')

#表示範囲の指定
plt.xlim(0, 3500000)
plt.ylim(0, 0.6)

#凡例の追加
plt.legend(ncol=2)

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/train/similarity_graph/compare")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/train/similarity_graph/compare/' + date1 + '.png')

#グラフの表示
plt.show()

print("Showing Similarity Compare graph is completed.")