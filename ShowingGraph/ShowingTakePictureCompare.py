import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date1 = "23-09-03 02-46-00"
date2 = "23-08-27 21-24-26"
mode = "val"

df1 = pd.read_csv("log/" + date1 + "/" + mode + "/metrics.csv", names=['time', 'picture', 'ci', 'exp_area', 'path_length'], header=None)
plt.plot(df1['time'], df1['picture'], color="blue", label="Base")
df2 = pd.read_csv("log/" + date2 + "/" + mode + "/metrics.csv", names=['time', 'picture', 'ci', 'exp_area', 'path_length'], header=None)
plt.plot(df2['time'], df2['picture'], color="red", label="Penalty")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Number of Times Taken Pictures')

#表示範囲の指定
plt.xlim(0, 10000000)
plt.ylim(0, 100)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/taken_picture_graph/compare")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/taken_picture_graph/compare/' + date1 + '.png')

#グラフの表示
plt.show()

print("Showing Taken Picture graph is completed.")