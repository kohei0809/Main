import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-08-01 19-24-19"
date = "23-08-06 16-55-35"
date = "23-08-06 17-02-58"
mode = "train"
mode = "val"

df = pd.read_csv("log/" + date + "/" + mode + "/metrics.csv", names=['time', 'ci', 'episode_length', 'exp_area', 'path_length'], header=None)
plt.plot(df['time'], df['episode_length'], color="blue", label="Episode Length")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Episode Length')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 1.0)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/episode_length_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/episode_length_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Episode Length graph is completed.")