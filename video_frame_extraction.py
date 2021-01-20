# -*- coding:utf8 -*-
import cv2
import os
import shutil
from tqdm import tqdm


def get_frame_from_video(video_name, interval, save_path):
    """
    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间隔
    Returns:
    """

    # 保存图片的路径
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            # 保存图片
            j += 1
            save_name = save_path + '/' + str(j) + '_' + str(i) + '.jpg'
            if frame is not None:
                cv2.imwrite(save_name, frame)
                # print('image of %s is saved' % save_name)
            else:
                print(f"{save_name} is None")
        if not success:
            print('video is all read')
            break


if __name__ == '__main__':
    # 视频文件名字
    data_path = "./datasets/人脸反欺诈数据集/CASIA-FASD/CASIA_faceAntisp/test_release"
    interval = 10
    save_path = './images/val_data/CASIA-FASD/CASIA_faceAntisp/test_release'
    for person in tqdm(os.listdir(data_path)):
        if "DS_Store" in person:
            continue
        data_person_path = os.path.join(data_path, person)
        for video in tqdm(os.listdir(data_person_path)):
            if "DS_Store" in person:
                continue
            video_name = os.path.join(data_person_path, video)
            video_save_path = os.path.join(save_path, video_name.rsplit('/', 2)[1], video_name.rsplit('/', 2)[2].split('.')[0])
            get_frame_from_video(video_name, interval, video_save_path)

'''
每个主体下的文件夹对应标签
1，2：真实人体
3，4：A4照片攻击
5，6：照片攻击，眼部裁剪掉
7，8：Ipad照片攻击
HR_1: 高清真实人体
HR_2: 高清A4照片攻击
HR_3: 高清照片攻击，眼部裁剪掉
HR_4: 高清Ipad照片攻击
'''