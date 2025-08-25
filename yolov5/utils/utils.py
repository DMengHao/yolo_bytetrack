# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision.transforms import Resize
import time
from pathlib import Path
import sys
from torchvision.ops import box_iou

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def save_pic(image, save_path):
    cv2.imencode('.jpg', image)[1].tofile(save_path)


def draw_pic(img_read, text, x1, y1, x2, y2):
    cv2.rectangle(img_read, (x1, y1), (x2, y2), (0, 0, 220), 2)
    cv2.putText(img_read, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)


def draw(img_tensor, detections, img_names, image_name):
    save_path = ROOT /'results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, img in enumerate(img_tensor):
        image = img.cpu().numpy().astype(np.uint8)
        if os.path.splitext(img_names[i])[1] in ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.svg', '.pfg']:
            for item in detections[i]:
                text = item['class']
                box = item['box']
                conf = item['conf']
                label = f'{text} ({conf:.2f})'
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                draw_pic(image, label, x1, y1, x2, y2)  # 使用方法打包汇框和标签，便于维护
            pic_save_path = f'{save_path}/{image_name+os.path.basename(img_names[i])}'
            save_pic(image, pic_save_path)
            print(f'保存推理结果图路径：{pic_save_path}')


def IoU(box1, box2) -> float:
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union

def nms_task1(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 7680
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 0.2375ms vs 0.1187ms
        # idxs = torch.arange(boxes.shape[0], device=boxes.device)
        # i = torchvision.ops.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output

def byte_track_nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
        labels=(), max_det=300):
    nc = prediction.shape[2] - 5  # 类别数量（总列数 - 5个基础字段）
    xc = prediction[..., 4] > conf_thres  # 初步过滤：目标置信度>阈值的候选框

    assert 0 <= conf_thres <= 1, f'置信度阈值{conf_thres}无效，需在0.0-1.0之间'
    assert 0 <= iou_thres <= 1, f'IOU阈值{iou_thres}无效，需在0.0-1.0之间'

    min_wh, max_wh = 2, 7680  # 过滤极小/极大框的宽高阈值
    max_nms = 30000  # NMS前最大框数量
    time_limit = 10.0  # 超时限制（秒）
    redundant = True
    multi_label &= nc > 1  # 多标签模式仅在多类别时有效
    merge = False  # 是否合并重叠框

    t = time.time()
    # 初始化输出：每个样本对应一个空张量（7列，对应目标格式）
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):
        # 过滤宽高异常的框（设置其目标置信度为0）
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        # 应用初步过滤（目标置信度>阈值）
        x = x[xc[xi]]

        # 若有标签，合并标签到检测结果中（用于训练/验证，推理时可忽略）
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # 边界框
            v[:, 4] = 1.0  # 目标置信度
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # 类别置信度
            x = torch.cat((x, v), 0)

        if not x.shape[0]:  # 无有效框，跳过
            continue

        # 保存原始目标置信度（obj_conf）和类别置信度（class_conf）
        obj_conf = x[:, 4:5].clone()  # 目标存在置信度（第5列）
        class_conf_original = x[:, 5:].clone()  # 原始类别置信度（未乘目标置信度）
        combined_conf = obj_conf * class_conf_original  # 综合置信度（用于过滤和NMS）

        # 转换边界框格式：xywh -> xyxy（左上角+右下角）
        box = xywh2xyxy(x[:, :4])

        # 处理多标签/单标签模式
        if multi_label:
            # 多标签：保留所有置信度>阈值的类别
            i, j = (combined_conf > conf_thres).nonzero(as_tuple=False).T  # i是框索引，j是类别索引
            # 拼接：(x1,y1,x2,y2, obj_conf, class_conf, class_pred)
            x = torch.cat((
                box[i],  # 边界框
                obj_conf[i],  # 目标置信度
                class_conf_original[i, j].unsqueeze(1),  # 类别置信度
                j[:, None].float()  # 类别ID
            ), 1)
        else:
            # 单标签：每个框保留置信度最高的类别
            conf_combined, j = combined_conf.max(1, keepdim=True)  # 最大综合置信度及类别索引
            class_conf = class_conf_original.gather(1, j)  # 对应类别的原始类别置信度
            # 拼接并过滤：仅保留综合置信度>阈值的框
            x = torch.cat((
                box,  # 边界框
                obj_conf,  # 目标置信度
                class_conf,  # 类别置信度
                j.float()  # 类别ID
            ), 1)[conf_combined.view(-1) > conf_thres]

        # 过滤指定类别（如只保留"人"）
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:  # 过滤后无有效框，跳过
            continue
        # 限制NMS前的最大框数量（避免计算量过大）
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按目标置信度排序

        # NMS：按类别分组（避免不同类别互相抑制）
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # 类别偏移（不同类别框坐标偏移，避免跨类别NMS）
        boxes, scores = x[:, :4] + c, x[:, 4] * x[:, 5]  # NMS用的框和综合置信度（用于排序）
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # 执行NMS

        # 限制输出框数量
        if i.shape[0] > max_det:
            i = i[:max_det]

        # 合并重叠框（可选，用于提升密集目标精度）
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres  # 计算重叠度
            weights = iou * scores[None]  # 权重=重叠度×置信度
            # 合并框坐标（加权平均）
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]  # 保留被多个框覆盖的目标

        # 保存当前样本的结果
        output[xi] = x[i]

        # 检查是否超时
        if (time.time() - t) > time_limit:
            print(f'警告：NMS超时（超过{time_limit}秒）')
            break

    return output

def nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 7680
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 0.2375ms vs 0.1187ms
        # idxs = torch.arange(boxes.shape[0], device=boxes.device)
        # i = torchvision.ops.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output

def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) # 计算输入图像相较于原始图像的缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def letterbox_xu(im, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, stride=32): # im->(B,3,1280,1280)
    shape = im.shape[2:]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #四舍五入，取整
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        torch_resize = Resize([new_unpad[1], new_unpad[0]])
        im = torch_resize(im)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    p2d = (left, right, top, bottom)
    out = F.pad(im, p2d, 'constant', 114.0 / 255.0)
    out = out.contiguous()
    return out, ratio, (dw, dh)

def draw(img, detections, save_path, color_map):
    image = img
    if os.path.splitext(save_path)[1] in ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.svg', '.pfg']:
        for item in detections:
            text = item['class']
            box = item['box']
            conf = item['conf']
            label = f'{text} ({conf:.2f})'
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            color = color_map.get(text, [np.random.randint(0, 255) for _ in range(2)])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2)
        save_pic(image, save_path)

def get_images_tensor(path):
    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    tensor_data = []
    image_path_list = os.listdir(path)
    all_image_list = [os.path.join(path,i) for i in image_path_list]
    for i, image_path in enumerate(all_image_list):
        img_read = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img_read = torch.from_numpy(img_read).float()
        img_read = img_read.to(device)
        tensor_data.append(img_read)
    return tensor_data, image_path_list