import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date1 = "23-10-26 18-29-56"
date2 = "23-10-26 19-48-15"

mode = "train"
#mode = "eval"

df_reward1 = pd.read_csv("log/" + date1 + "/" + mode + "/reward.csv", names=['time', 'reward'], header=None)
df_metrics1 = pd.read_csv("log/" + date1 + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length'], header=None)
plt.plot(df_reward1['time'], df_reward1['reward'], color="red", label="Picture_reward")
plt.plot(df_metrics1['time'], df_metrics1['ci'], color="blue", label="Picture_CI")
plt.plot(df_metrics1['time'], df_metrics1['exp_area'], color="green", label="Picture_area")
plt.plot(df_metrics1['time'], df_reward1['reward']-df_metrics1['ci']-df_metrics1['exp_area'], color="black", label="Picture_distance")

df_reward2 = pd.read_csv("log/" + date2 + "/" + mode + "/reward.csv", names=['time', 'reward'], header=None)
df_metrics2 = pd.read_csv("log/" + date2 + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length'], header=None)
plt.plot(df_reward2['time'], df_reward2['reward'], "--", color="red", label="Movie_reward")
plt.plot(df_metrics2['time'], df_metrics2['ci'], "--", color="blue", label="Movie_CI")
plt.plot(df_metrics2['time'], df_metrics2['exp_area'], "--", color="green", label="Movie_area")
plt.plot(df_metrics2['time'], df_reward2['reward']-df_metrics2['ci']-df_metrics2['exp_area'], "--", color="black", label="Movie_distance")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Reward')

#表示範囲の指定
plt.xlim(0, 1200000)
plt.ylim(0, 50)

#凡例の追加
plt.legend(ncol=2)

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/reward_graph/each/compare")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/reward_graph/each/compare/' + date1 + '.png')

#グラフの表示
plt.show()

print("Showing Each Reward Compare graph is completed.")