import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(images_name, model_dir, device_id, threshold, model_test=None):
    if not model_test:
        model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    video_name = images_name.rsplit('/', 1)[1].rsplit('_', 1)[0]
    res = []
    for image in os.listdir(images_name):
        if "DS_Store" in image:
            continue
        img = cv2.imread(os.path.join(images_name, image))
        # result = check_image(image)
        # if result is False:
        #     return
        image_bbox = model_test.get_bbox(img)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": img,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        # draw result of prediction
        threshold = float(threshold)
        if prediction[0][1] >= 2 * threshold:
            label = 1
        else:
            label = np.argmax(prediction)

        value = prediction[0][label]/2

        res.append([label, value, image, os.path.join(images_name, image), image_bbox])

    positive_res = []
    negative_res = []
    for e in res:
        # print(e[0], e[1], e[2])
        if e[0] == 1:
            positive_res.append(e)
        else:
            negative_res.append(e)

    output = None
    if len(res) <= len(positive_res) * 2:
        positive_res.sort(key=lambda x: x[1], reverse=True)
        output = positive_res[0]
        print("video '{}' is Real Face. Best Score: {:.2f}. Best frame: {}.".format(video_name, output[1], output[2]))
        result_text = "RealFace Score: {:.2f}".format(output[1])
        color = (255, 0, 0)
    else:
        negative_res.sort(key=lambda x: x[1], reverse=True)
        output = negative_res[0]
        print("video '{}' is Fake Face. Score: {:.2f}. Best frame: {}.".format(video_name, output[1], output[2]))
        result_text = "FakeFace Score: {:.2f}".format(output[1])
        color = (0, 0, 255)

    print("Anti-spoofing prediction cost {:.2f} s".format(test_speed))

    new_img = cv2.imread(output[3])
    cv2.rectangle(
        new_img,
        (output[4][0], output[4][1]),
        (output[4][0] + output[4][2], output[4][1] + output[4][3]),
        color, 2)
    cv2.putText(
        new_img,
        result_text,
        (output[4][0], output[4][1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*new_img.shape[0]/1024, color)

    result_image_name = 'result_' + output[2]
    cv2.imwrite(os.path.join(images_name, result_image_name), new_img)

    return output[0], os.path.join(output[3])


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
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
        "--image_name",
        type=str,
        default="./datasets/video_data/1-1.avi",
        help="image used to test")
    args = parser.parse_args()
    label = test(args.image_name, args.model_dir, args.device_id)
