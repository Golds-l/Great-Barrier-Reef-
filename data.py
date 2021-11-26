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
              dstPath="D:\\kaggle"):
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
        shutil.copy(imagePath + "\\" + "\\".join(pt.split("\\")[-2:]).replace("txt", "jpg"),
                    dstImagePath + f"\\{fileName}.jpg")


def generateYAML(path="D:\\kaggle\\kaggle", nc=1, names=None):
    if names is None:
        names = ["starfish"]
    if nc != len(names):
        raise "NUM OF CLASSES ERROR!"
    if not os.path.exists(f"{path}\\images\\train"):
        raise f"{path}\\images\\train NOT EXISTS!"
    path = path[:-1] if path.endswith("\\") else path
    fileName = path + "\\" + path.split("\\")[-1] + ".yaml"
    with open(fileName, "w") as y:
        y.write(f"train: {path}\\images\\train\\\nval: {path}\\images\\val\\\nnc: {nc}\nnames: {names}")


def divide(imagePath, labelPath, stride, savePath):
    """
    :param savePath: path\\to\\save
    :param imagePath: a glob path
    :param labelPath: a glob path, too
    :param stride:
    :return: num of val
    """
    if savePath.endswith("\\"):
        savePath = savePath[:-1]
    os.mkdir(savePath + "\\" + "images")
    os.mkdir(savePath + "\\" + "labels")
    os.mkdir(savePath + "\\" + "images\\" + "train")
    os.mkdir(savePath + "\\" + "images\\" + "val")
    os.mkdir(savePath + "\\" + "labels\\" + "train")
    os.mkdir(savePath + "\\" + "labels\\" + "val")
    numVal = 0
    imgPaths = glob.glob(imagePath)
    labelPaths = glob.glob(labelPath)
    for pathNum in range(len(labelPaths)):
        if not os.path.exists(labelPaths[pathNum].replace("labels", "images").replace("txt", "jpg")):
            print(labelPaths[pathNum].replace("labels", "images").replace("txt", "jpg") + "  No such file or directory")
            continue
        if pathNum % stride == 0:
            shutil.copyfile(labelPaths[pathNum],
                            f"{savePath}\\labels\\val\\" + labelPaths[pathNum].split("\\")[-1])
            shutil.copyfile(labelPaths[pathNum].replace("labels", "images").replace("txt", "jpg"),
                            f"{savePath}\\images\\val\\" + labelPaths[pathNum].split("\\")[-1].replace("txt", "jpg"))
            numVal += 1
        else:
            shutil.copyfile(labelPaths[pathNum],
                            f"{savePath}\\labels\\train\\" + labelPaths[pathNum].split("\\")[-1])
            shutil.copyfile(labelPaths[pathNum].replace("labels", "images").replace("txt", "jpg"),
                            f"{savePath}\\images\\train\\" + labelPaths[pathNum].split("\\")[-1].replace("txt", "jpg"))
    return numVal


if __name__ == "__main__":
    # divide("D:\\kaggle\\trainData\\images\\*.jpg", "D:\\kaggle\\trainData\\labels\\*.txt", 8, "D:\\kaggle\\yoloData")
    generateYAML()
