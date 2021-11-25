import glob
import os
import shutil

import pandas as pd
import cv2


def showImages(csvPath="D:\\kaggle\\tensorflow-great-barrier-reef\\train.csv",
               fps=30,
               imagePath="D:\\kaggle\\tensorflow-great-barrier-reef\\train_images"):
    labels = pd.read_csv(csvPath)
    for idx in range(labels.shape[0]):
        imageLabels = eval(labels.iloc[idx, 5])
        imgName = labels.iloc[idx, 4].split("-")
        if len(imageLabels):
            print(f"video_{imgName[0]}\\{imgName[1]}.jpg")
            img = cv2.imread(
                f"{imagePath}\\video_{imgName[0]}\\{imgName[1]}.jpg")
            for lbl in imageLabels:
                imgDraw = cv2.rectangle(img, (lbl["x"], lbl["y"]), (lbl["x"] + lbl["width"], lbl["y"] + lbl["height"]),
                                        (255, 0, 0), 2)
                cv2.imshow("imgDraw", imgDraw)
                cv2.waitKey(0 if fps == 0 else (1000 // fps))


def makeLabels(csvPath="D:\\kaggle\\tensorflow-great-barrier-reef\\train.csv"):
    labels = pd.read_csv(csvPath)
    for idx in range(labels.shape[0]):
        imageLabels = eval(labels.iloc[idx, 5])
        imgName = labels.iloc[idx, 4].split("-")
        if len(imageLabels):
            with open(f".\\labels\\video_{imgName[0]}\\{imgName[1]}.txt", "a") as t:
                for lbl in imageLabels:
                    t.write(
                        f"0 {round((lbl['x'] + lbl['width'] / 2) / 1280, 7)} {round((lbl['y'] + lbl['height'] / 2) / 720, 7)} {round(lbl['width'] / 1280, 7)} {round(lbl['height'] / 720, 7)}\n")
                print(f"video_{imgName[0]}\\{imgName[1]}.jpg")


def moveFiles(txtPath='D:\\Projects\\kaggle\\labels\\*\\*.txt',
              imagePath="D:\\kaggle\\tensorflow-great-barrier-reef\\train_images",
              dstPath="D:\\kaggle\\tensorflow-great-barrier-reef"):
    txtPaths = glob.glob(txtPath)
    dstPath = dstPath[:-1] if dstPath.endswith("\\") else dstPath
    dstTxtPath = dstPath + "\\trainData\\labels"
    dstImagePath = dstPath + "\\trainData\\images"
    if not os.path.exists(dstTxtPath) or not os.path.exists(dstImagePath):
        try:
            os.makedirs(dstTxtPath)
            os.makedirs(dstImagePath)
        except FileExistsError as e:
            print("Folder Exists")
    for idx, pt in enumerate(txtPaths):
        fileName = pt.split("\\")[-2].split("_")[-1] + "_" + pt.split("\\")[-1].split(".")[0]
        shutil.copy(pt, dstTxtPath + f"\\{fileName}.txt")
        shutil.copy(imagePath + "\\" + "\\".join(pt.split("\\")[-2:]).replace("txt", "jpg"), dstImagePath + f"\\{fileName}.jpg")


if __name__ == "__main__":
    showImages(fps=0)
