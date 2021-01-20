import cv2
from face_recognition import FaceRecognition
import time

def test(data_path, gpu_id, nms, face_db, threshold, face_recognitio=None):

    # init model
    if not face_recognitio:
        face_recognitio = FaceRecognition(gpu_id, nms, face_db, threshold)

    img = cv2.imread(data_path)
    start = time.time()
    results = face_recognitio.recognition(img)
    end = time.time()
    print("Face recognition prediction cost {:.2f} s".format(end-start))

    if len(results) == 0:
        print("没有检测到人脸")
    else:
        for result in results:
            if result['user_id'] == 'unknown':
                flag = True
                if flag:
                    user_id = face_recognitio.register(img, results)
                    print(f"注册成功，用户ID：{user_id}，用户名：{data_path}")
                    return '注册'
            else:
                print(f"认证成功，用户ID为：{results[0]['user_id']}，用户名：{data_path}")
                return '认证'

