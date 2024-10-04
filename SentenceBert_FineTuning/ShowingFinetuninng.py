import pandas as pd
import matplotlib.pyplot as plt
import pathlib

mode = ""
mode = "2"
mode = "3"
mode = "4"
shuffle = True
#shuffle = False
add_word = ""

if shuffle == True:
    add_word = "_shuffle"

train_path = f"train{mode}{add_word}.csv"
train_df = pd.read_csv(train_path, names=["time", "loss"], header=None)
test_path = f"test{mode}{add_word}.csv"
test_df = pd.read_csv(test_path, names=["time", "loss"], header=None)

plt.plot(train_df['time'], train_df['loss'], color="red", label="Train")
plt.plot(test_df['time'], test_df['loss'], color="blue", label="Test")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Loss')

#表示範囲の指定
plt.xlim(0, 10000)
plt.ylim(1, 2)

#凡例の追加
plt.legend()

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig(f'./result/loss{mode}{add_word}.png')

#グラフの表示
plt.show()

print("Showing Loss graph is completed.")