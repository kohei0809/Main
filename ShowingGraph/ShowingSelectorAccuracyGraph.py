import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "24-03-13 04-09-55"

df = pd.read_csv("PictureSelector/log/" + date + "/train_removal_network.csv", names=['time', 'loss', 'accuracy'], header=None)
plt.plot(df['time'], df['accuracy'], color="blue", label="Accuracy")


#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Accuracy')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 10)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("PictureSelector/result/accuracy_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('PictureSelector/result/accuracy_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Selector Accuracy graph is completed.")