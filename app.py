from flask import Flask, request, render_template, jsonify
import os
import json
import io
import PIL.Image as Image
import numpy as np
import requests
from pymongo import MongoClient
from pymongo import DESCENDING
from datetime import datetime, timedelta
import time
import random
import pytz

import argparse
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


filename = 0
weightFile = 0
meal = 0
lastTotalWeight = 0
currentTotalWeight = 0

client = MongoClient("mongodb://127.0.0.1:27017")
print("Connection Successful")
client.drop_database('foodDB')
db = client['foodDB']
collection = db['data']

app = Flask(__name__)
app.secret_key = 'key'

# Food detection global variable
Detected_label = []
current_food = 'pizza'
Food_label = ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']
detectnew = 0

current_weights = 0

# get data of image from esp8266, save image to local
# in process_get(), food is recognized by model and data is stored in the mongodb
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global path, filename
    if request.method == 'POST':
        content = request.get_data()
        for i in range(len(content) - 1):
            if (content[i:i + 2] == b'\xff\xd9'):
                content = content[:i + 2]
        img = Image.open(io.BytesIO(b'\xff' + content))
        path = os.path.join(app.config['UPLOAD_FOLDER'], str(filename) + '.jpeg')
        img.save(path)
        # filename += 1
        print(path)
        process_get()
        return "upload image"

    if request.method == 'GET':
        image = [i for i in os.listdir('static/images') if i.endswith('.jpeg')][-1]
        return render_template("image.html", image=image)


# get the measured weight from the esp8266
# calculate the weight of the new added item
# store the weight in the variable lastTotalWeight
@app.route('/weight', methods=['POST'])
def weight():
    global path, weightFile, lastWeight, current_weights
    if request.method == 'POST':
        currentTotalWeight = float(request.get_data())
        weight = currentTotalWeight - lastTotalWeight
        lastTotalWeight = currentTotalWeight
        print(weight)
        return "upload weight"


# used for debugging
# insert food data into mongoDB
@app.route('/upload', methods=['POST'])
def upload():
    # foodjson = {'foods':[{'name': 'banana', 'weight': 20, 'calories': 1020}, {'name': 'apple', 'weight': 30, 'calories': 1000}], 'meal': 1, 'date':'2022.12.8'}
    data = json.loads(request.get_data())
    collection.insert_one(data)
    cursor = collection.find()
    data_display = []
    for d in cursor:
        data_display.append(d)
    return "upload"


# upload food name, weight, calories, mealID, date to mongoDB
def uploadToDB():
    global current_food, current_weights
    foodjson = {'foods': {'name': current_food, 'weight': current_weights, \
                          'calories': get_cal(current_food, current_weights)}, 'meal': get_meal(), 'date': get_date()}
    data = json.loads(json.dumps(foodjson))
    collection.insert_one(data)
    cursor = collection.find()
    data_display = []
    for d in cursor:
        data_display.append(d)
    print(data_display)


# get the total calories for today
# used for pie char on the android dashboard page
@app.route('/caltoday', methods=['GET'])
def caltoday():
    date = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
    cursor = collection.find({'date': date})
    totalCal = 0
    for item in cursor:
        totalCal += item['foods']['calories']
    return str(totalCal)


# get the total calories for this week
# used for bar char on the android dashboard page
@app.route('/calweek', methods=['GET'])
def calweek():
    today = datetime.now(pytz.timezone("America/New_York"))
    weekCal = {}
    for i in range(1, 8):
        day = today - timedelta(days=(7 - i))
        date = day.strftime("%Y-%m-%d")
        cursor = collection.find({'date': date})
        totalCal = 0
        for item in cursor:
            totalCal += item['foods']['calories']
        weekCal[str(i)] = totalCal
    return jsonify(weekCal)


# get food names, weights, calories for the current meal
# this will be used to display data on the android home page
@app.route('/meal', methods=['GET'])
def food():
    cursor = collection.find({"meal":get_meal()})
    data = []
    for item in cursor:
        food = item['foods']
        data.append(food)

    return jsonify({"foods":data})

# use nutritionix api to get nutrition facts label for a specific kind of food
# calculate the unit calories (KCal/gram) based on the nutrition facts label
# compute the calories for the food based on unit calories and weight
def get_cal(food, weights):
    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    item = str(food)
    apiId = "3017f827"
    apiKey = "744c4142f3c41e14db7f4f9caaa4b863"

    response = requests.post(url, data={"query": item}, headers={"x-app-id": apiId, "x-app-key": apiKey})

    resp_json = response.json()
    cal = resp_json["foods"][0]["nf_calories"]
    weight = resp_json["foods"][0]["serving_weight_grams"]
    unit_cal = cal / weight
    total_cal = unit_cal * weights
    return total_cal

# attach mealID for each meal 
def get_meal():
    timestamp = int(time.time())
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    h = t[11:13]
    if int(h) > 5 and int(h) <= 10:
        c_meal = 1
    elif int(h) > 10 and int(h) < 16:
        c_meal = 2
    else:
        c_meal = 3
    return c_meal

# get the date of today
def get_date():
    today = datetime.now(pytz.timezone("America/New_York"))
    date = today.strftime("%Y-%m-%d")
    return date


# food categories detection
# this part is reference from YOLOv5: https://github.com/ultralytics/yolov5
# only newly detected food lable will be stored
@smart_inference_mode()
def detect(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'static/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global current_food, Detected_label, Food_label, detectnew
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    print(names[int(c)])
                    # only newly detected food category will be recorded
                    if names[int(c)] in Food_label:
                        if names[int(c)] not in Detected_label:
                            current_food = names[int(c)]
                            Detected_label.append(names[int(c)])
                            detectnew = 1
                            print("new food", names[int(c)])
                            break
                        else:
                            detectnew = 0
                            current_food = ''
                    else:
                        detectnew = 0
                        current_food = ''
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print("s is:",s)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def process_get():
    global detectnew,current_food,Detected_label
    detect()
    if detectnew >0:
        print("detected new food: ", current_food)
        uploadToDB()
    else:
        print("there is no new food in the picture")
        print("previous food is: ", Detected_label)


if __name__ == "__main__":
    UPLOAD_FOLDER = '/home/ubuntu/yolov5/static/images'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(host='0.0.0.0', port=8080)


