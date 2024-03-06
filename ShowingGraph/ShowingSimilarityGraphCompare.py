import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date1 = "24-02-18 15-24-41"
date2 = "24-02-20 18-05-03"
date3 = "24-02-21 23-05-39"
date4 = "24-02-24 06-09-40"

df1 = pd.read_csv("log/" + date1 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'path_length'], header=None)
plt.plot(df1['time'], df1['similarity'], color="blue", label="method 1")
df2 = pd.read_csv("log/" + date2 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'path_length'], header=None)
plt.plot(df2['time'], df2['similarity'], color="red", label="method 2")
df3 = pd.read_csv("log/" + date3 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'path_length'], header=None)
plt.plot(df3['time'], df3['similarity'], color="green", label="method 3")
df4 = pd.read_csv("log/" + date4 + "/train/metrics.csv", names=['time', 'ci', 'exp_area', 'similarity', 'path_length'], header=None)
plt.plot(df4['time'], df4['similarity'], color="black", label="method 2 (random selection)")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Sentence Similarity')

#表示範囲の指定
plt.xlim(0, 5000000)
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