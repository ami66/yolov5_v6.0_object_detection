
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def convert(img0,img_size=640, stride=32, auto=True):
    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0


def load_model(device='cpu', weights='./weight/fry_best.pt', data_conf='./data/custom_data.yaml', dnn=False, half=False, imgsz=(640, 640)):
    # Load model
    device = select_device(device)
    ini_model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data_conf)
    #stride, names, pt, jit, onnx, engine = ini_model.stride, ini_model.names, ini_model.pt, ini_model.jit, ini_model.onnx, ini_model.engine

    # Half
    model = ini_model
    half &= (ini_model.pt or ini_model.jit or ini_model.onnx or ini_model.engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if ini_model.pt or ini_model.jit:
        model.model.half() if half else model.model.float()

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

    return model,ini_model


@torch.no_grad()
def predict(
        image,
        model,
        ini_model,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        ):

    # Dataloader
    #cudnn.benchmark = True  # set True to speed up constant image size inference

    im, im0s=convert(image,img_size=imgsz, stride=ini_model.stride, auto=ini_model.pt)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=augment, visualize=visualize)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 =im0s.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            res=[]
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label =ini_model.names[c]
                xyxy=[str(int(x)) for x in xyxy]
                #print('坐标，类别，置信度：',xyxy,label,conf)
                xyxy.append(label)
                res.append(xyxy)
    return res

if __name__ == "__main__":


    device='cpu' #如果是使用GPU，则填入int型的0,1,2,3
    path = './test/img/395edfd78b51bfcc2bfde6386a860589.jpeg' #图片路径
    weight_path='./weight/fry_best.pt' #模型文件路径
    data_conf='./data/custom_data.yaml' #数据集配置文件路径

    img0 = cv2.imread(path)
    model,ini_model=load_model(device=device,weights=weight_path, data_conf=data_conf)
    result=predict(image=img0,model=model,ini_model=ini_model,device=device)
    print(result)
