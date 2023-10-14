import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns 
import numpy as np

date = "23-08-08 21-46-33"
mode = "val"

df = pd.read_csv("check/CI/score3.csv", header=None)
print(df.max().max())
print(df.min().min())
plt.figure()
sns.heatmap(df, vmin=-1.0, vmax=5.0, cmap='jet', cbar=False)
ax = plt.subplot(1, 1, 1)
ax.axis("off")
ax.set_aspect('equal')
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/heat_map")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/heat_map/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Exp Area graph is completed.")