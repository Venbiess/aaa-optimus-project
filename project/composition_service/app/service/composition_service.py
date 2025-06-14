import torch
import numpy as np
import cv2
import os
import json

from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model
from iharm.mconfigs import ALL_MCONFIGS
from app.service.bbox_calculation import compute_bbox_coordinates
from app.service.compose import gaussian_composite_image


class CompositionService:
    def __init__(self):
        """
        Initializes the CompositionService by loading CNN and ViT harmonization models
        and all available background images.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # loading both models
        model_cnn = load_model('CNN_pct', './pretrained_models/PCTNet_CNN.pth', verbose=False)
        use_attn = ALL_MCONFIGS['CNN_pct']['params']['use_attn']
        normalization = ALL_MCONFIGS['CNN_pct']['params']['input_normalization']
        self.predictor_cnn = Predictor(model_cnn, device, use_attn=use_attn, mean=normalization['mean'], std=normalization['std'])

        model_vit = load_model('ViT_pct', './pretrained_models/PCTNet_ViT.pth', verbose=False)
        use_attn = ALL_MCONFIGS['ViT_pct']['params']['use_attn']
        normalization = ALL_MCONFIGS['ViT_pct']['params']['input_normalization']
        self.predictor_vit = Predictor(model_vit, device, use_attn=use_attn, mean=normalization['mean'], std=normalization['std'])

        # loading the preset backgrounds
        image_folder = 'app/images/backgrounds'
        self.backgrounds = dict()

        # Read all .png files from the folder
        for filename in os.listdir(image_folder):
            if filename.lower().endswith('.png'):
                img_path = os.path.join(image_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.backgrounds[filename.split('.')[0]] = img.astype(np.uint8)
        
        with open(image_folder + '/bbox_config.json', 'r') as f:
            self.bbox_config = json.load(f)


    def compose(self, foreground:np.array, mask:np.array, background_id:int=1, model_type:str='vit'):
        """
        Composes the foreground image onto a background using a binary mask and harmonizes it.

        Args:
            foreground (np.array): Foreground image in OpenCV format.
            mask (np.array): Binary mask of the foreground.
            background_id (int): Index of the background to use (1-based).
            model_type (str): Type of harmonization model ('none', 'cnn' or 'vit').

        Returns:
            np.array: The harmonized composite image.
        """
        # getting the background
        background = self.backgrounds[str(background_id)]
        # calcuating the bbox
        bbox_conf = self.bbox_config.get(str(background_id), dict())
        bbox = compute_bbox_coordinates(mask, background,
                                        **bbox_conf)
        # blending the image
        comp_img, comp_mask = gaussian_composite_image(background, foreground, mask, bbox, kernel_size=5)
        # harmonizing the foreground
        comp_mask = (comp_mask > 127).astype(np.uint8)
        comp_lr = cv2.resize(comp_img, (256, 256))
        mask_lr = cv2.resize(comp_mask, (256, 256))
        if model_type == 'none':
            pred_img = comp_img
        elif model_type == 'cnn':
            _, pred_img = self.predictor_cnn.predict(comp_lr, comp_img, mask_lr, comp_mask)
        elif model_type == 'vit':
            _, pred_img = self.predictor_vit.predict(comp_lr, comp_img, mask_lr, comp_mask)

        return pred_img
