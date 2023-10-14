from PIL import Image, ImageDraw
import numpy as np
import pathlib
import matplotlib.pyplot as plt

#各種設定
scene_name = "759xd9YjKW5"
number = 0

filename = "./map_ci/" + scene_name + "_" + str(number) + ".csv"        
        
#filenameのデータを配列に格納
data = np.loadtxt(filename, dtype="float", delimiter= ",")
#print(data)
        
#配列dataのサイズの画像を作成
height, width = data.shape
img = Image.new("RGB", (width, height))
        
#格納されている数字に従って画像の色を設定
for y in range(height):
    for x in range(width):
        ci = data[y][x]
        #探索不可能領域(壁等)は黒
        if ci == -100:
            img.putpixel((x, y), (0, 0, 0))
        
        #探索可能領域だがデータ不足は白
        elif ci == -200:
            img.putpixel((x, y), (255, 255, 255))
            
        #ciが負の場合は青色
        elif ci < 0:
            img.putpixel((x, y), (0, 0, 255))
            
        #ciが正の場合はだんだん赤にしていく
        elif ci < 2:
            img.putpixel((x, y), (255, 255, 0))
        
        elif ci < 4:
            img.putpixel((x, y), (255, 178, 178))
            
        elif ci < 6:
            img.putpixel((x, y), (255, 127, 127))
            
        elif ci < 8:
            img.putpixel((x, y), (255, 76, 76))
            
        elif ci < 10:
            img.putpixel((x, y), (255, 25, 25))
            
        elif ci >= 10:
            img.putpixel((x, y), (255, 0, 0))
        
        else:
            print("({0},{1})でエラー".format(x,y))
    
#フォルダがない場合は、作成 
p_dir = pathlib.Path("./result/ci_map")
if not p_dir.exists():
    p_dir.mkdir(parents=True)
    
#画像を保存
img.save("./result/ci_map/ci_map_" + scene_name + "_" + str(number) + ".png")

print(scene_name + "_" + str(number) + " CI Map completed.")