import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date1 = "23-12-22 23-13-05"
date2 = "24-01-08 12-14-22"
date3 = "24-01-13 12-21-17"
mode = "train"

df1 = pd.read_csv("log/" + date1 + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length'], header=None)
plt.plot(df1['time'], df1['ci'], color="blue", label="Individual")
df2 = pd.read_csv("log/" + date2 + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length'], header=None)
plt.plot(df2['time'], df2['ci'], color="red", label="Object Category Map")
df3 = pd.read_csv("log/" + date3 + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length', 'object_num'], header=None)
plt.plot(df3['time'], df3['ci'], color="green", label="2-Scale Map")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('$CI$')

#表示範囲の指定
plt.xlim(0, 2500000)
plt.ylim(0, 50)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/ci_graph/compare")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/ci_graph/compare/' + date1 + '.png')

#グラフの表示
plt.show()

print("Showing CI graph compare is completed.")