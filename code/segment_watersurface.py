import os
import numpy as np
import cv2
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
from PIL import Image

CONFIG_FILE = 'config/Segformer.py'
CHECKPOINT_FILE = 'config/Segformer.pth'

DEVICE = 'cuda:0'

PALETTE = {
    0: [0, 0, 0],
    1: [0, 255, 0]
}

opacity = 0.5

model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

def create_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def my_predict_water(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Unable to read image at {img_path}")
        return

    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in PALETTE.items():
        pred_mask_bgr[pred_mask == idx] = color

    pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0)

    folder = "./database/" + os.path.basename(img_path).split(".")[0] + '/seg_img'
    create_output_folder(folder)

    cv2.imwrite(os.path.join(folder, 'seg.jpg'), pred_mask_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred_viz_rgb = cv2.cvtColor(pred_viz, cv2.COLOR_BGR2RGB)

    original_image = Image.fromarray(img_rgb)
    pred_viz_image = Image.fromarray(pred_viz_rgb)
    blended_image = Image.blend(original_image, pred_viz_image, alpha=0.3)

    blended_image.save(os.path.join(folder, 'seg_blend.jpg'))