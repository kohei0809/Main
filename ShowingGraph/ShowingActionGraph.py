import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

date = "24-10-05 21-07-23"
date = "24-10-07 06-13-55"
date = "24-10-02 01-50-40"
date = "24-10-07 19-11-01"
mode = "train5"
#mode = "train"

df = pd.read_csv(f"log/{date}/{mode}/action_prob.csv", names=['time', 'forward', 'left', 'right', 'look_up', 'look_down', 'take_picture',], header=None)
#plt.plot(df['time'], df['take_picture'], color="red", label="Take Picture")
plt.plot(df['time'], df['forward'], color="red", label="Move Forward")
plt.plot(df['time'], df['left'], color="blue", label="Turn Left")
plt.plot(df['time'], df['right'], color="green", label="Turn Right")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Action Probability')

#表示範囲の指定
#plt.xlim(0, 1750000)
plt.ylim(0, 1.0)

#plt.xticks(np.arange(0, 2000000, 400000))

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/action_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/action_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Action graph is completed.")