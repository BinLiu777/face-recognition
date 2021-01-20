import os
import yaml
import numpy as np
import insightface
import cv2
from sklearn import preprocessing
import logging
import time

# logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.DEBUG)

class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            deploy_conf = configs["FACE"]
            # 正数为GPU的ID，负数为使用CPU
            self.gpu_id = deploy_conf["GPU_ID"]
            self.face_db = deploy_conf["FACE_DB"]
            self.threshold = deploy_conf["THRESHOLD"]
            self.nms = deploy_conf["NMS"]

class FaceRecognition:
    def __init__(self, gpu_id, nms, face_db, threshold, val=False, detection=False):
        # 加载人脸识别模型
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=gpu_id, nms=nms)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.maxID = 0
        self.threshold = threshold
        self.face_db = face_db
        self.nms = nms
        self.gpu_id = gpu_id
        if not val and not detection:
            self.load_faces(face_db)


    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                if file.endswith('bin'):
                    dtype = file.split('.')[0].split('-')[1]
                    user_id = file.split('.')[0].split('-')[0]
                    if int(user_id) > self.maxID:
                        self.maxID = int(user_id)
                    logging.info("加载人脸：%s" % user_id)
                    embedding = np.fromfile(os.path.join(root, file),dtype=dtype)

                self.faces_embedding.append({
                    "user_id": user_id,
                    "feature": embedding
                })

    def face_detection(self, image, save_path):
        save_path, name = save_path.rsplit("/",1)[0], save_path.rsplit("/",1)[1].split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        faces = self.model.get(image)
        results = 0
        for face in faces:
            # 获取人脸属性
            bbox = np.array(face.bbox).astype(np.int32).tolist()
            landmarks = np.array(face.landmark).astype(np.int32).tolist()
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
            for (x, y) in landmarks:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), 2)
            save_path = save_path.split('.')[0]
            cv2.imwrite(save_path + '/' + name + '-detection.jpg', image)
            results += 1
        return results

    def recognition(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["landmark"] = np.array(face.landmark).astype(np.int32).tolist()
            result["age"] = face.age
            gender = '男'
            if face.gender == 0:
                gender = '女'
            result["gender"] = gender
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["user_id"] = "unknown"
            result["embedding"] = embedding
            similar_res = []
            for com_face in self.faces_embedding:
                logging.info("和%s进行对比" % com_face["user_id"])
                r, dist = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    similar_res.append((dist, com_face["user_id"]))
            if similar_res: # 排序选出距离最小的人
                similar_res.sort()
                result["user_id"] = similar_res[0][1]
                result["similar_Users"] = similar_res
            results.append(result)
        return results

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        logging.info("人脸欧氏距离：%f" % dist)
        if dist < threshold:
            return True, dist
        else:
            return False, dist

    def register(self, image, faces):
        # faces = self.model.get(image)
        if len(faces) < 1:
            print("没有检测到人脸，无法注册")
            return None
        # 判断人脸是否存在
        face = faces[0]
        embedding = preprocessing.normalize(face['embedding'])

        user_id = self.maxID + 1
        self.maxID = user_id
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        embedding.tofile(os.path.join(self.face_db, '%s-%s.bin' % (user_id, embedding.dtype)))
        cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_id))
        self.faces_embedding.append({
            "user_id": user_id,
            "feature": embedding
        })
        return user_id

if __name__ == '__main__':
    data_path = 'Data/童星'

    # init model
    start_time = time.time()
    face_recognitio = FaceRecognition("config.yaml")
    finish_init_time = time.time()
    print('init model time: ', finish_init_time-start_time)

    for person in os.listdir(data_path):
        if person.endswith('DS_Store'):
            continue
        for age in os.listdir(data_path + '/' + person):
            if age.endswith('DS_Store'):
                continue
            for image in os.listdir(data_path + '/' + person + '/' + age):
                if image.endswith('DS_Store'):
                    continue
                if image.endswith('.bin'):
                    continue
                process_image_time = time.time()
                print("Process: "+image)
                img = cv2.imread(data_path + '/' + person + '/' + age + '/' + image)
                results = face_recognitio.recognition(img)
                if len(results) == 0:
                    print("没有检测到人脸")
                else:
                    for result in results:
                        if result['user_id'] == 'unknown':
                            flag = True
                            if flag:
                                user_id = face_recognitio.register(img, results)
                                print(f"注册成功，用户ID：{user_id}，用户名：{image}")
                        else:
                            print(f"认证成功，用户ID为：{results[0]['user_id']}，用户名：{image}")
                        finish_process_time = time.time()
                        print("process "+image+" time:", finish_process_time-process_image_time)
