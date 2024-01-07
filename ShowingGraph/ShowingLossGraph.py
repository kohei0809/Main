import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-12-22 23-13-05"

df = pd.read_csv("log/" + date + "/train/loss.csv", names=['time', 'loss_value', 'loss_policy'], header=None)
plt.plot(df['time'], df['loss_value'], color="red", label="loss_value")
plt.plot(df['time'], df['loss_policy'], color="blue", label="loss_policy")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Loss')

#表示範囲の指定
#plt.xlim(0, 5000000)
#plt.ylim(0, 4500000)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/train/loss_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/train/loss_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Loss graph is completed.")