# -*- coding:utf8 -*-

import os
import shutil
import argparse
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
        default="./datasets/video_data/1-3.avi",
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

    image_save_path = os.path.join(args.data_path.rsplit('/', 1)[0], args.data_path.rsplit('/', 1)[1].split('.')[0]+'_images')

    # 抽帧
    video_frame_extraction.get_frame_from_video(args.data_path, args.interval, image_save_path)

    # 活体检测
    label, image_path = anti_spoofing_test.test(image_save_path, args.model_dir, args.device_id, args.threshold_anti_spoofing, model_test)

    # 人脸注册和识别
    if label == 1:
        face_recognition_test.test(image_path, args.device_id, args.nms, args.face_db_path, args.threshold_recognition, face_recognitio)
    else:
        print('假脸攻击')
    shutil.rmtree(image_save_path)
