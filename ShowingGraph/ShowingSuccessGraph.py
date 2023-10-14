import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-07-18 12-02-14"

df = pd.read_csv("log/" + date + "/val/metrics.csv", names=['time', 'distance_to_currgoal', 'distance_to_multi_goal', 'episode_length', 'mspl', 'pspl', 'percentage_success', 'success', 'sub_success'], header=None)
plt.plot(df['time'], df['success'], color="blue", label="success")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Success')

#表示範囲の指定
plt.xlim(0, 50000000)
plt.ylim(0, 1.0)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/val/success_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/val/success_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Success graph is completed.")