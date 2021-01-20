# -*- coding:utf8 -*-

import os
import shutil
import argparse
from tqdm import tqdm
import video_frame_extraction
import anti_spoofing_test
import face_recognition_test
from src.anti_spoof_predict import AntiSpoofPredict
from face_recognition import FaceRecognition

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets/人脸反欺诈数据集/CASIA-FASD/CASIA_faceAntisp/test_release",
        help="image used to test")
    parser.add_argument(
        "--device_id",
        type=int,
        default=-1,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--threshold_anti_spoofing",
        type=str,
        default=0.1,
        help="threshold to judge positive, default = 0.1")
    parser.add_argument(
        "--threshold_recognition",
        type=str,
        default=1.24,
        help="threshold to judge positive, default = 1.24")
    parser.add_argument(
        "--l2tol1",
        type=str,
        default=False,
        help="label 2 belong to label 1, default = False")
    parser.add_argument(
        "--interval",
        type=str,
        default=10,
        help="interval for video to images")
    parser.add_argument(
        "--face_db_path",
        type=str,
        default="./face_db",
        help="face_db path")
    parser.add_argument(
        "--nms",
        type=str,
        default=0.5,
        help="")
    args = parser.parse_args()

    model_test = AntiSpoofPredict(args.device_id)
    face_recognitio = FaceRecognition(args.device_id, args.nms, args.face_db_path, args.threshold_recognition)

    true_predict = 0
    totol = 0
    wrong_case = []
    persons = set()

    # for person in tqdm(os.listdir(args.data_path), desc="Number of processed person"):
    for person in os.listdir(args.data_path):
        if "DS_Store" in person:
            continue
        for video in os.listdir(os.path.join(args.data_path, person)):
            if "DS_Store" in video:
                continue
            if "images" in video:
                continue

            data_path = os.path.join(args.data_path, person, video)
            print(f"Process {data_path}")

            image_save_path = os.path.join(data_path.rsplit('/', 1)[0], data_path.rsplit('/', 1)[1].split('.')[0]+'_images')

            # 抽帧
            video_frame_extraction.get_frame_from_video(data_path, args.interval, image_save_path)

            # 活体检测
            label, image_path = anti_spoofing_test.test(image_save_path, args.model_dir, args.device_id, args.threshold_anti_spoofing, model_test)

            # 人脸注册和识别
            if label == 1:
                res_label = face_recognition_test.test(image_path, args.device_id, args.nms, args.face_db_path, args.threshold_recognition, face_recognitio)
            else:
                print('假脸攻击')
                res_label = '假脸'
            shutil.rmtree(image_save_path)


            video_name = video.split('.')[0]
            if video_name == '1' or video_name == '2' or video_name == 'HR_1':  # 真实人面
                if person not in persons:
                    persons.add(person)
                    if res_label == '注册':
                        true_predict += 1
                    else:
                        wrong_case.append([data_path, res_label, '注册'])
                else:
                    if res_label == '认证':
                        true_predict += 1
                    else:
                        wrong_case.append([data_path, res_label, '认证'])
            else:
                if res_label == '注册' or res_label == '认证':
                    wrong_case.append([data_path, res_label, '假脸'])
                else:
                    true_predict += 1
            totol += 1
            print()

    acc = true_predict / totol
    print(acc)

    with open(os.path.join(args.data_path, f"wrong_case.txt"), "w") as f:
        for e in wrong_case:
            f.write(f"{e[0]}: label:{e[1]}, truth:{e[2]}\n")  # 自带文件关闭功能，不需要再写f.close()






