import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "24-02-18 15-24-41"
#date = "24-02-20 18-05-03"
#date = "24-02-21 23-05-39"
date = "24-06-14 15-28-02"
date = "24-07-25 06-34-14"
mode = "train3"

df1 = pd.read_csv(f"log/{date}/{mode}/reward.csv", names=['time', 'reward'], header=None)
df2 = pd.read_csv(f"log/{date}/{mode}/metrics.csv", names=['time', 'exp_area', 'similarity',  'picture_value', 'pic_sim', 'subgoal_reward', 'blue_score', 'rouge_1_score', 'rouge_2_score', 'rouge_L_score', 'meteor_score', 'path_length'], header=None)
plt.plot(df1['time'], df1['reward'], color="red", label="Reward")
plt.plot(df2['time'], df2['similarity'], color="blue", label="Similarity")
plt.plot(df2['time'], df2['exp_area'], color="green", label="Exp Area")
plt.plot(df2['time'], df2['subgoal_reward'], color="black", label="Sub-Goal Reward")

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