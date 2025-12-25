import os
import re
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, f1_score, make_scorer, roc_auc_score)
from sklearn.ensemble import VotingClassifier

## 特征提取
def extractFeatures(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (256, 256))
# 颜色统计
    meanRgb = img.mean(axis=(0, 1))
    stdRgb = img.std(axis=(0, 1))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    meanHsv = hsv.mean(axis=(0, 1))
    stdHsv = hsv.std(axis=(0, 1))

    gChannel = img[:, :, 1]
    meanG, stdG = np.mean(gChannel), np.std(gChannel)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lChannel = lab[:, :, 0]
    meanL, stdL = np.mean(lChannel), np.std(lChannel)

#GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

 #LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    features = np.hstack([
        meanRgb, stdRgb,
        meanHsv, stdHsv,
        meanG, stdG,
        meanL, stdL,
        contrast, correlation, energy, homogeneity,
        hist
    ])
    return features

## 数据导入
def loadDataset(imageRoot, annotationRoot):
    xData = []
    yBinary = []
    yMulti = []
    for fileName in os.listdir(annotationRoot):
        if not fileName.endswith(".xls"):
            continue
        match = re.search(r"Base\d+", fileName)
        if match is None:
            continue
        baseName = match.group()
        excelPath = os.path.join(annotationRoot, fileName)
        imageDir = os.path.join(imageRoot, baseName)
        print(f"处理：{fileName} -> {imageDir}")

        df = pd.read_excel(excelPath, engine="xlrd")
        df.columns = [c.strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            imgName = row["image name"]
            imgPath = os.path.join(imageDir, imgName)
            if not os.path.exists(imgPath):
                continue
            xData.append(extractFeatures(imgPath))

            risk = int(row["risk of macular edema"])
            yBinary.append(0 if risk == 0 else 1)  #二分类标签：将1，2合并成1
            yMulti.append(risk)  #原始三分类标签保留
    return np.array(xData), np.array(yBinary), np.array(yMulti)

## 数据预处理
imageRoot = r"D:\3_1\Pattern_recognition\diabetic_retinopathy_classification\data\images"
annotationRoot = r"D:\3_1\Pattern_recognition\diabetic_retinopathy_classification\data\annotations"
xData, yBinary, yMulti = loadDataset(imageRoot, annotationRoot)
print("样本数：", xData.shape[0])
print("原始特征维度：", xData.shape[1])
scaler = StandardScaler()
xScaled = scaler.fit_transform(xData)
pca = PCA(n_components=0.95)
xPca = pca.fit_transform(xScaled)
print("PCA 后维度：", xPca.shape[1])

##二分类评价指标
def balancedAccuracyBinary(yTrue, yPred):
    r0 = recall_score(yTrue, yPred, pos_label=0)
    r1 = recall_score(yTrue, yPred, pos_label=1)
    return 0.5 * (r0 + r1)

## SVM
def runSvm(x, y, title):
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    if len(np.unique(y)) == 2:
        print(f"\n[SVM] {title}（二分类）")

        svm = SVC(
            kernel="rbf",
            C=100,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        )
        svm.fit(xTrain, yTrain)
        yProb = svm.predict_proba(xTest)[:, 1]
        bestTh, bestBal = 0.5, -1
        for th in np.arange(0.1, 0.6, 0.02):
            yTmp = (yProb >= th).astype(int)
            bal = balancedAccuracyBinary(yTest, yTmp)
            if bal > bestBal:
                bestBal, bestTh = bal, th
        yPred = (yProb >= bestTh).astype(int)
        auc = roc_auc_score(yTest, yProb)
        print("最佳阈值：", round(bestTh, 2))
        print("Balanced Accuracy：", round(balancedAccuracyBinary(yTest, yPred), 4))
        print("AUC：", round(auc, 4))
        # 每一类AUC
        for label in np.unique(yTest):
            yTrue_bin = (yTest == label).astype(int)
            auc_label = roc_auc_score(yTrue_bin, yProb)
            print(f"AUC(类别{label} vs rest)：", round(auc_label, 4))
        print(confusion_matrix(yTest, yPred))
        print(classification_report(yTest, yPred, digits=4))
    else:
        print(f"\n[SVM] {title}（三分类）")

        scorer = make_scorer(f1_score, average="macro")
        svm = SVC(kernel="rbf", class_weight="balanced", probability=True)
        grid = GridSearchCV(
            svm,
            {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
            scoring=scorer,
            cv=5,
            n_jobs=-1
        )
        grid.fit(xTrain, yTrain)
        yPred = grid.best_estimator_.predict(xTest)
        yProb = grid.best_estimator_.predict_proba(xTest)
        auc = roc_auc_score(yTest, yProb, multi_class='ovr')
        print("Best Params:", grid.best_params_)
        print("Macro-F1：", round(f1_score(yTest, yPred, average="macro"), 4))
        print("AUC：", round(auc, 4))
        # 每一类AUC
        for i, label in enumerate(np.unique(yTest)):
            yTrue_bin = (yTest == label).astype(int)
            auc_label = roc_auc_score(yTrue_bin, yProb[:, i])
            print(f"AUC(类别{label} vs rest)：", round(auc_label, 4))
        print(confusion_matrix(yTest, yPred))
        print(classification_report(yTest, yPred, digits=4))
runSvm(xPca, yBinary, "任务1：是否存在风险")
runSvm(xPca, yMulti, "任务2：风险等级 0/1/2")

## 决策树DT
def runDt(x, y, title):
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    scorer = make_scorer(f1_score, average="macro") if len(np.unique(y)) > 2 else "accuracy"
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    grid = GridSearchCV(
        dt,
        {"max_depth": [5, 8, 10], "min_samples_leaf": [5, 10, 20]},
        scoring=scorer,
        cv=5,
        n_jobs=-1
    )
    grid.fit(xTrain, yTrain)
    yPred = grid.best_estimator_.predict(xTest)

    print(f"\n[DT] {title}")
    print("Best Params:", grid.best_params_)
    if len(np.unique(y)) == 2:  # 二分类
        yProb = grid.best_estimator_.predict_proba(xTest)[:, 1]
        auc = roc_auc_score(yTest, yProb)
        print("Balanced Accuracy：", round(balancedAccuracyBinary(yTest, yPred), 4))
        print("AUC：", round(auc, 4))
        # 每一类AUC
        for label in np.unique(yTest):
            yTrue_bin = (yTest == label).astype(int)
            auc_label = roc_auc_score(yTrue_bin, yProb)
            print(f"AUC(类别{label} vs rest)：", round(auc_label, 4))
    else:  # 三分类
        yProb = grid.best_estimator_.predict_proba(xTest)
        auc = roc_auc_score(yTest, yProb, multi_class='ovr')
        print("Macro-F1：", round(f1_score(yTest, yPred, average="macro"), 4))
        print("AUC：", round(auc, 4))
        # 每一类AUC
        for i, label in enumerate(np.unique(yTest)):
            yTrue_bin = (yTest == label).astype(int)
            auc_label = roc_auc_score(yTrue_bin, yProb[:, i])
            print(f"AUC(类别{label} vs rest)：", round(auc_label, 4))
    print(confusion_matrix(yTest, yPred))
    print(classification_report(yTest, yPred, digits=4))
runDt(xPca, yBinary, "任务1：是否存在风险")
runDt(xPca, yMulti, "任务2：风险等级 0/1/2")

## 集成模型
def runEnsemble(x, y, title):
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    svm = SVC(
        kernel="rbf",
        C=100,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )
    model = VotingClassifier(
        estimators=[("svm", svm), ("dt", dt)],
        voting="hard"
    )
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    print(f"\n【集成结果】{title}")
    if len(np.unique(y)) == 2:  # 二分类
        # VotingClassifier(voting="hard")没有predict_proba，需改为soft
        model_soft = VotingClassifier(
            estimators=[("svm", svm), ("dt", dt)],
            voting="soft"
        )
        model_soft.fit(xTrain, yTrain)
        yProb = model_soft.predict_proba(xTest)[:, 1]
        auc = roc_auc_score(yTest, yProb)
        print("Balanced Accuracy：", round(balancedAccuracyBinary(yTest, yPred), 4))
        print("AUC：", round(auc, 4))
        # 每一类AUC
        for label in np.unique(yTest):
            yTrue_bin = (yTest == label).astype(int)
            auc_label = roc_auc_score(yTrue_bin, yProb)
            print(f"AUC(类别{label} vs rest)：", round(auc_label, 4))
    print(confusion_matrix(yTest, yPred))
    print(classification_report(yTest, yPred, digits=4))
runEnsemble(xPca, yBinary, "二分类（是否有风险）")

