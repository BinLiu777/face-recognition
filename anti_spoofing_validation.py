# TPR：在所有实际为阳性的样本中，被正确地判断为阳性之比率。TPR=TP/(TP+FN)
# FPR：在所有实际为阴性的样本中，被错误地判断为阳性之比率。FPR=FP/(FP+TN)

# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoofing_test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/val_data/"

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
# def check_image(image):
#     height, width, channel = image.shape
#     if width/height != 3/4:
#         print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
#         return False
#     else:
#         return True


def val(val_data_path, model_dir, device_id, threshold, l2tol1):

    TP = 0  # 检测是positive，实际也是positive
    FP = 0  # 检测是positive，实际也是negative
    TN = 0  # 检测是negative，实际是negative
    FN = 0  # 检测是negative，实际是negative

    wrong_case = []

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    val_data_path = SAMPLE_IMAGE_PATH + val_data_path

    if "NUAA" in val_data_path:

        positive_data_path = val_data_path + 'ClientRaw/'
        negative_data_path = val_data_path + 'ImposterRaw/'
        positive_id_data_path = val_data_path + 'client_test_raw.txt'
        negative_id_data_path = val_data_path + 'imposter_test_raw.txt'

        mark_positive = True

        data_path = [positive_data_path, negative_data_path]
        data_id_path = [positive_id_data_path, negative_id_data_path]
        for i in range(2):
            if i == 1:
                mark_positive = False
            if i == 0:
                mark_positive = True

            with open(data_id_path[i], "r") as f:
                print(f"Process: {data_id_path[i]}")
                for line in tqdm(f.readlines()):
                    line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    line = line.replace('\\','/')
                    image_path = data_path[i] + line

                    image = cv2.imread(image_path)

                    # result = check_image(image)
                    # if result is False:
                    #     print('未检测到人脸')
                    #     return

                    image_bbox = model_test.get_bbox(image)
                    prediction = np.zeros((1, 3))
                    val_speed = 0
                    # sum the prediction from single model's result
                    for model_name in os.listdir(model_dir):
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": image,
                            "bbox": image_bbox,
                            "scale": scale,
                            "out_w": w_input,
                            "out_h": h_input,
                            "crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img = image_cropper.crop(**param)
                        # start = time.time()
                        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                        # val_speed += time.time()-start

                    # draw result of prediction
                    # print(prediction) # [[0.62197654 1.02370903 0.35431439]]

                    if threshold is not None:
                        threshold = float(threshold)
                        if prediction[0][1] >= 2*threshold:
                            label = 1
                        else:
                            label = np.argmax(prediction)
                    else:
                        label = np.argmax(prediction)

                    if mark_positive:
                        if l2tol1:
                            if label == 1 or label == 2:
                                TP += 1
                            else:
                                wrong_case.append([val_data_path, label, 1])
                                FN += 1
                        else:
                            if label == 1:
                                TP += 1
                            else:
                                wrong_case.append([val_data_path, label, 1])
                                FN += 1
                    else:
                        if l2tol1:
                            if label == 1 or label == 2:
                                wrong_case.append([val_data_path, label, 0])
                                FP += 1
                            else:
                                TN += 1
                        else:
                            if label == 1 or label == 2:
                                wrong_case.append([val_data_path, label, 0])
                                FP += 1
                            else:
                                TN += 1

                    value = prediction[0][label] / 2

                wrong_case.append([])

    elif "CASIA" in val_data_path:

        for person in tqdm(os.listdir(val_data_path), desc='process person'):
            if "DS_Store" in person:
                continue
            if '.txt' in person:
                continue
            path2 = os.path.join(val_data_path, person)
            for video in os.listdir(path2):
                if "DS_Store" in person:
                    continue
                path3 = os.path.join(path2, video)
                for image in os.listdir(path3):
                    if "DS_Store" in person:
                        continue
                    path4 = os.path.join(path3, image)
                    image = cv2.imread(path4)

                    image_bbox = model_test.get_bbox(image)
                    prediction = np.zeros((1, 3))
                    val_speed = 0
                    # sum the prediction from single model's result
                    for model_name in os.listdir(model_dir):
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": image,
                            "bbox": image_bbox,
                            "scale": scale,
                            "out_w": w_input,
                            "out_h": h_input,
                            "crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img = image_cropper.crop(**param)
                        # start = time.time()
                        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                        # val_speed += time.time()-start

                    # draw result of prediction
                    # print(prediction) # [[0.62197654 1.02370903 0.35431439]]

                    if threshold is not None:
                        threshold = float(threshold)
                        if prediction[0][1] >= 2*threshold:
                            label = 1
                        else:
                            label = np.argmax(prediction)
                    else:
                        label = np.argmax(prediction)

                    if video == '1' or video == '2' or video == 'HR_1': # 真实人面
                        if l2tol1:
                            if label == 1 or label == 2:
                                TP += 1
                            else:
                                wrong_case.append([path4, label, 1])
                                FN += 1
                        else:
                            if label == 1:
                                TP += 1
                            else:
                                wrong_case.append([path4, label, 1])
                                FN += 1
                    else:
                        if l2tol1:
                            if label == 1 or label == 2:
                                wrong_case.append([path4, label, 0])
                                FP += 1
                            else:
                                TN += 1
                        else:
                            if label == 1:
                                wrong_case.append([path4, label, 0])
                                FP += 1
                            else:
                                TN += 1
                    value = prediction[0][label] / 2

                wrong_case.append([])

    print(TP, FP, TN, FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_score = 2 / ((1 / precision) + (1 / recall))
    TPR = recall
    FPR = FP / (FP + TN)
    ROC = TPR / FPR

    with open(os.path.join(val_data_path, f"wrong_case_{threshold if threshold is not None else 'None'}_{'l2tol1' if l2tol1 else 'None'}.txt"), "w") as f:
        for e in wrong_case:
            if len(e) == 0:
                f.write('\n')
            else:
                f.write(f"{e[0]}: label:{e[1]}, truth:{e[2]}\n")  # 自带文件关闭功能，不需要再写f.close()

        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"F1_score: {F1_score}\n")
        f.write(f"TPR: {TPR}\n")
        f.write(f"FPR: {FPR}\n")
        f.write(f"ROC: {ROC}\n")

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"accuracy: {accuracy}")
    print(f"F1_score: {F1_score}")
    print(f"TPR: {TPR}")
    print(f"FPR: {FPR}")
    print(f"ROC: {ROC}")


if __name__ == "__main__":
    desc = "val"
    parser = argparse.ArgumentParser(description=val)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="CASIA-FASD/CASIA_faceAntisp/test_release/",
        # default="NUAA/raw/",
        help="image used to test")
    parser.add_argument(
        "--threshold",
        type=str,
        default=0.1,
        help="threshold to judge positive, default = None")
    parser.add_argument(
        "--l2tol1",
        type=str,
        default=False,
        help="label 2 belong to label 1, default = False")
    args = parser.parse_args()
    val(args.val_data_path, args.model_dir, args.device_id, args.threshold, args.l2tol1)
