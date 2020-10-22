#import os
from pathlib import *
import torch
import subprocess
from yolov3.models import Darknet
from yolov3.utils.utils import non_max_suppression, rescale_boxes
from yolov3.utils.datasets import pad_to_square, resize

def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


#use pathlib for better supporting windows
class YOLOv3:
    def __init__(
            self,
            device,
            img_size=416,
            person_detector=False,
            video=False,
            return_dict=False
    ):

       # homedir = os.path.expanduser("~")
        homedir = Path.home()

        #weights_path = os.path.join(homedir, '.torch/models/yolov3.weights')
        weights_path = homedir.joinpath('.torch/models/yolov3.weights')
       
        #os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        weights_path.parent.mkdir( parents=False, exist_ok=True)

        if not weights_path.exists():
            url = 'https://pjreddie.com/media/files/yolov3.weights'
            outdir = weights_path.parent
            download_url(url, outdir)

        model_def = homedir.joinpath('.torch/config/yolov3.cfg')
        model_def.parent.mkdir( parents=False, exist_ok=True)

        if not model_def.exists():
            url = 'https://raw.githubusercontent.com/mkocabas/yolov3-pytorch/master/yolov3/config/yolov3.cfg'
            outdir = model_def.parent
            download_url(url, outdir)

        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.img_size = img_size
        self.video = video
        self.person_detector = person_detector
        self.device = device
        self.return_dict = return_dict

        self.model = Darknet(model_def, img_size=img_size).to(device)
        self.model.load_darknet_weights(weights_path)
        # self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, batch):
        if self.video:
            inp_batch = []
            for img in batch:
                # Pad to square resolution
                img, _ = pad_to_square(img, 0)
                # Resize
                img = resize(img, self.img_size)
                inp_batch.append(img)
            inp_batch = torch.stack(inp_batch).float().to(self.device)
        else:
            inp_batch = batch

        detections = self.model(inp_batch)
        detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        for idx, det in enumerate(detections):
            if det is None:
                det = {
                    'boxes': torch.empty(0,4),
                    'scores': torch.empty(0),
                    'classes': torch.empty(0),
                }
                detections[idx] = det
                continue

            if self.video:
                det = rescale_boxes(det, self.img_size, batch.shape[-2:])

            if self.person_detector:
                det = det[det[:,6] == 0]

            if self.return_dict:
                det = {
                    'boxes': det[:, :4],
                    'scores': det[:, 4] * det[:, 5],
                    'classes': det[:, 6],
                }

            detections[idx] = det



        return detections
