import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

for i in range(90):
    df = pd.read_csv("research_ci_and_similarity/csv/ci_similarity_" + str(i) + ".csv", names=['ci', 'similarity'], header=None)
    plt.scatter(df['ci'], df['similarity'])
    
    # x軸の範囲を取得
    x_min, x_max = plt.xlim()

    # y軸に点線を引く
    for xi in np.arange(np.ceil(x_min), np.floor(x_max) + 1):
        plt.axvline(x=xi, linestyle='dashed', color='gray')


    #ラベルの追加
    plt.xlabel('CI')
    plt.ylabel('Similarity')

    #表示範囲の指定
    #plt.xlim(0, 50000000)
    #plt.ylim(0, 1.0)

    #指数表記から普通の表記に変換
    plt.ticklabel_format(style='plain',axis='x')
    plt.ticklabel_format(style='plain',axis='y')
    

    #フォルダがない場合は、作成
    p_dir = pathlib.Path("./research_ci_and_similarity/scatter")
    if not p_dir.exists():
        p_dir.mkdir(parents=True)

    #グラフの保存
    plt.savefig('./research_ci_and_similarity/scatter/' + str(i) + '.png')

    #グラフの表示
    plt.show()
    
    plt.clf()

print("Showing CI and Similarity Scatter is completed.")