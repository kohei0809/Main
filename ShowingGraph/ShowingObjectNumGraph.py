import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "24-01-13 12-21-17"

mode = "train"
#mode = "eval"

df = pd.read_csv("log/" + date + "/" + mode + "/metrics.csv", names=['time', 'ci', 'exp_area', 'distance', 'path_length', 'object_num'], header=None)
plt.plot(df['time'], df['object_num'], color="blue", label="Observed Objects")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Number of Observed Objects')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 1.0)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/object_num_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/object_num_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Object Num graph is completed.")