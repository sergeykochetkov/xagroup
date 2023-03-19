import os
import cv2
from argparse import ArgumentParser
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
from mmdet3d.apis.inference import show_proj_det_result_meshlab


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('videos_dir', help='path to videos directory')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    tmp_dir = 'tmp'
    show_every = 5
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = os.path.join(tmp_dir, 'tmp.png')

    model = init_model(args.config, args.checkpoint, device=args.device)
    ann_file = 'ann.json'

    for filename in os.listdir(args.videos_dir):
        cap = cv2.VideoCapture(os.path.join(args.videos_dir, filename))

        count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:

                if count % show_every == 0:
                    cv2.imwrite(image_path, frame)
                    result, data = inference_mono_3d_detector(model, image_path, ann_file)
                    file_name = show_proj_det_result_meshlab(data, result, tmp_dir,
                                                             score_thr=model.test_cfg.score_thr, show=True,
                                                             snapshot=False)
                    #cv2.imshow("frame", frame)
                    #cv2.waitKey(1)
                count += 1
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
