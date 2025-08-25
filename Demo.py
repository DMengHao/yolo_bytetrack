import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
from loguru import logger
from bytetrack.utils.visualize import plot_tracking
from bytetrack.tracker.byte_tracker import BYTETracker
from yolov5.yolov5_trt import YOLOv5_TRT


class yolo_bytetrack:
    def __init__(self, yolo_api, traker):
        self.yolo_api = yolo_api
        self.traker = traker
    
    def infer(self, image):
        outputs, img_info = self.yolo_api.infer(image)
        results = []
        if outputs[0] is not None:
                online_targets = self.traker.update(outputs[0], [img_info['height'], img_info['width']], (img_info['height'], img_info['width'])) # 更新追踪器,根据检测结果和图像尺寸，更新目标追踪状态
                online_tlwhs = [] # 存储目标边界框 左上角，宽，高
                online_ids = [] # 存储目标ID
                for t in online_targets:
                    tlwh = t.tlwh # 获取边界框
                    tid = t.track_id # 获取追踪ID
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh # 过滤掉高宽比异常（过窄或过高）和面积较小的框
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        temp = ([int(data) for data in tlwh.tolist()])
                        temp[2] = temp[0] + temp[2]
                        temp[3] = temp[1] + temp[3]
                        temp.extend([tid])
                        results.append(temp)
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids)
        else: # 若未检测到目标，直接使用原图
            online_im = img_info['raw_img']
        return online_im, results
    

def imageflow_demo(video_path, vis_folder, current_time, yolo_bytetrack_api):
    # 初始化视频捕获设备
    cap = cv2.VideoCapture(video_path)
    # 获取视频基本信息
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 结果保存路径
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp) # 结果保存路径
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    # 视频写入器初始化
    vid_writer = cv2.VideoWriter( # 初始化视频写入器，保存路径，编码格式，帧率，分辨率
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    frame_id = 0 
    results = [] 
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} fps)'.format(frame_id))
        ret_val, frame = cap.read()
        if ret_val:
            online_im, box_id = yolo_bytetrack_api.infer(frame)
            if True: # 若开启保存结果，将绘制后的帧写入视频
                vid_writer.write(online_im)
            ch = cv2.waitKey(1) # 监听键盘输入：按ESC/q/Q退出
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if True: # 循环结束后，若开启保存功能将追踪结果写入文本文件
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(args):      
    # vis_folder = './results'
    os.makedirs(args.save_path, exist_ok=True)
    current_time = time.localtime()
    yolov5_api = YOLOv5_TRT(weights=r'E:\DMH\Shanghai\ByteTrack-main\tools\yolov5\weights\xf.engine', dev='cuda:0', det_type='xf')
    tracker = BYTETracker(args, frame_rate=240) # 初始化ByteTracker追踪器
    yolo_bytetrack_api = yolo_bytetrack(yolo_api=yolov5_api, traker=tracker)
    imageflow_demo(args.video_path, args.save_path, current_time, yolo_bytetrack_api)


def parse_opt(known=False):
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--video_path", type=str, default=r"E:\DMH\Shanghai\ByteTrack-Demo\video\temp.mp4", help="测试视频路径")
    parser.add_argument("--save_path", type=str, default=r"E:\DMH\Shanghai\ByteTrack-Demo\results", help="视频结果保存路径")
    # 目标跟踪参数配置
    parser.add_argument("--track_thresh", type=float, default=0.5, help="目标跟踪时的置信度阈值")
    parser.add_argument("--track_buffer", type=int, default=30, help="跟踪缓存帧数")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="跟踪时目标匹配阈值")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,help="异常检测框过滤，宽高比")
    parser.add_argument('--min_box_area', type=float, default=10, help='过滤面积过小的检测框')
    parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="是否针对MOT20数据集调整跟踪策略，启用后会优化遮挡目标的跟踪逻辑")
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == "__main__":
    args = parse_opt(True)
    main(args)