import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "24-10-05 21-07-23"
date = "24-10-07 06-13-55"
date = "24-10-02 01-50-40"
date = "24-10-07 19-11-01"
mode = "train5"
#mode = "train"

df = pd.read_csv(f"log/{date}/{mode}/reward.csv", names=['time', 'reward'], header=None)
plt.plot(df['time'], df['reward'], color="blue", label="reward")

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
p_dir = pathlib.Path("./result/" + mode + "/reward_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/reward_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Reward graph is completed.")