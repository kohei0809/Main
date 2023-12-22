import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-10-26 18-29-56"
date = "23-12-11 22-34-18"
#date = "23-10-09 16-02-33"

mode = "train"
#mode = "eval"

df1 = pd.read_csv("log/" + date + "/" + mode + "/reward.csv", names=['time', 'reward'], header=None)
df2 = pd.read_csv("log/" + date + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length'], header=None)
plt.plot(df1['time'], df1['reward'], color="red", label="reward")
plt.plot(df2['time'], df2['ci'], color="blue", label="CI")
plt.plot(df2['time'], df2['exp_area'], color="green", label="EXP_AREA")
plt.plot(df2['time'], df2['distance'], color="black", label="distance")
#plt.plot(df2['time'], df1['reward']-df2['ci']-df2['exp_area'], color="black", label="distance")
#plt.plot(df2['time'], df2['distance'], color="black", label="distance")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Reward')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 10)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/reward_graph/each")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/reward_graph/each/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Each Reward graph is completed.")