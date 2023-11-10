import pandas as pd
import matplotlib.pyplot as plt
import pathlib


date1 = "23-10-26 18-29-56"
date2 = "23-10-26 19-48-15"

mode = "train"
#mode = "eval"

df1 = pd.read_csv("log/" + date1 + "/" + mode + "/reward.csv", names=['time', 'reward'], header=None)
plt.plot(df1['time'], df1['reward'], color="blue", label="Picture")
df2 = pd.read_csv("log/" + date2 + "/" + mode + "/reward.csv", names=['time', 'reward'], header=None)
plt.plot(df2['time'], df2['reward'], color="red", label="Movie")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Reward')

#表示範囲の指定
plt.xlim(0, 1200000)
plt.ylim(0, 40)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/reward_graph/compare")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/reward_graph/compare/' + date1 + '.png')

#グラフの表示
plt.show()

print("Showing Reward graph compare is completed.")