# -*- coding:utf-8 -*-
import os
import time
import logging
import cv2
import numpy as np
import torch
import tensorrt as trt
from pycuda import driver
import pycuda.driver as cuda0
from collections import OrderedDict, namedtuple
from tqdm import tqdm
from .utils.utils import letterbox_xu,nms,scale_coords,save_pic,get_images_tensor,draw,byte_track_nms
import pycuda.autoinit
import yaml
import os.path as osp

# 创建日志记录器并设置级别
logger = logging.getLogger('crossingdemo')
logger.setLevel(logging.INFO)
# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个控制台输出的处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 添加处理器
logger.addHandler(console_handler)

# with open('./utils/default.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# points = np.array(config['crossing1_area'])

class YOLOv5_TRT:
    def __init__(self, weights='./weights/new_data_20250222.engine', det_type='task', imgsz=1280, dev='cuda:0',
                 conf_thresh=0.25, iou_thresh=0.45, max_det=200, half=True):
        self.imgsz = imgsz
        self.device = int(dev.split(':')[-1])
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.half = half
        self.stride = 64 if self.imgsz == 1280 else 32
        self.det_type = det_type
        self.ctx = cuda0.Device(self.device).make_context()
        self.stream = driver.Stream()
        self.names = []
        if self.det_type == 'task':
            self.names = ['h','o','k','b']
        if self.det_type == 'xf':
            self.names = ['bird','person','train','animal', 'stone', 'yw', 'human', 'flood', 'mudslide', 'tree', 'float', 'square', 'box', 'light', 'cat', 'dog','traintop']
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        for index in range(self.model.num_bindings):
            if trt.__version__ >= '8.6.1':
                name = self.model.get_tensor_name(index)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(index)
                dtype = trt.nptype(self.model.get_binding_dtype(index))
                shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def preprocess(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().to(self.device)

        temp = (torch.flip(img, dims=[2]).permute(2, 0, 1)).unsqueeze(0)
        im = letterbox_xu(temp, self.imgsz,auto=False, stride=self.stride)[0]
        img_shape = (img.shape[0], img.shape[1])
        im_shape = (im.shape[2], im.shape[3])

        im = torch.divide(im, 255.0)
        im = im.half() if self.half else im.float()
        return im, img_shape, im_shape
    

    def postprocess(self, pred, img_list, im_list, img):
        
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = 1.0001

        pred = byte_track_nms(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, max_det=self.max_det)
        for k, det in enumerate(pred):
            H, W = img_list
            det[:, :4] = scale_coords(im_list, det[:, :4], img_list).round()
            coords = det[:, :4].long()
            coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, W - 1)
            coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, H - 1)
        
        return pred, img_info
    


        # detections = []
        # for k, det in enumerate(pred):
        #     H, W = img_list
        #     det[:, :4] = scale_coords(im_list, det[:, :4], img_list).round()
        #     coords = det[:, :4].long()
        #     coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, W - 1)
        #     coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, H - 1)

        #     classes = det[:, 5]
        #     confs = det[:, 4]

        #     detection = []
        #     for x1, y1, x2, y2, conf, cls in zip(*coords.T, confs, classes):
        #         cls_str = self.names[int(cls)]
        #         detection.append({'id': int(cls),
        #                           'class': cls_str,
        #                           'conf': float(conf),
        #                           'box': [x1.item(), y1.item(), x2.item(), y2.item()]})
        #     detections.append(detection)
        # return detections

    def infer(self, tensor_data):
        try:
            self.ctx.push()
            process_start_time = time.time()
            input_data, img_shape, im_shape = self.preprocess(tensor_data)
            logger.info(f'前处理用时：{(time.time() - process_start_time) * 1000:.4f}ms')
            infer_start_time = time.time()
            self.binding_addrs['images'] = int(input_data.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            logger.info(f'推理用时：{(time.time() - infer_start_time) * 1000:.4f}ms')
            post_start_time = time.time()
            preds = self.bindings['output'].data
            detections, img_info = self.postprocess(preds, img_shape, im_shape, tensor_data)
            # detections, img_info = self.postprocess(preds, tensor_data)
            logger.info(f'后处理用时：{(time.time() - post_start_time) * 1000:.4f}ms')
            return detections, img_info
        finally:
            self.ctx.pop()


    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


if __name__ == '__main__':

    ENGINE_PATH = r"E:\DMH\Shanghai\ByteTrack-main\yolov5\weights\xf.engine"
    color_map = {'o': (0, 0, 255)}
    IMAGE_PATH = r"E:\DMH\Shanghai\ByteTrack-main\yolov5\images"  # 待检测图片路径
    OUTPUT_PATH = r"E:\DMH\Shanghai\ByteTrack-main\yolov5\results"  # 输出图片路径
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    yolov5_api = YOLOv5_TRT(weights=ENGINE_PATH, det_type='xf')
    image_path = os.listdir(IMAGE_PATH)
    print('开始测试：')
    for i,img_path in tqdm(enumerate(image_path), total=len(image_path)):
        image = cv2.imdecode(np.fromfile(os.path.join(IMAGE_PATH,img_path), dtype=np.uint8), -1)
        results = yolov5_api.infer(image)
        draw(image, results[0], os.path.join(OUTPUT_PATH,img_path), color_map)
    print('测试完毕！')