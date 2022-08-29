import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np

from models import *
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import cv2
import torch

from evaluation import *
from utils import *
from postprocess import *

class CAM(object):
    def __init__(self, args, exp_name, encoder_model_type, gpuid):
        # encoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "encoder.pth")
        # decoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "decoder.pth")
        model_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "model.pth")

        self.model_model_type = encoder_model_type
        if encoder_model_type == "Res18":
            model = Res18_Classifier().cuda()
        elif encoder_model_type == "Res50":
            model = Res50_Classifier().cuda()
        model.load_pretrain_weight(model_path)
        # state_dict_weights = torch.load(encoder_path)
        # model.load_state_dict(state_dict_weights, strict=False)
        # state_dict_weights = torch.load(decoder_path)
        # model.load_state_dict(state_dict_weights, strict=False)
        for param in model.parameters():
            param.requires_grad = False

        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        self.model = model.to('cuda')

        self.result_path = os.path.join(args.project_path, "results_wacv", exp_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def step(self, img):
        img = img.cuda()
        feat_maps, feature, logit = self.model(img)
        pred = torch.sigmoid(torch.flatten(logit))
        return feat_maps.detach().cpu(), pred.detach().cpu(), logit.detach().cpu(), feature.detach().cpu()

    def get_cam_weight(self):
        weight = None
        for (k, v) in self.model.state_dict().items():
            if "decoder.0.weight" in k:
                weight = v.detach().cpu()
        return weight.squeeze(0).numpy()


    def run_selected_case(self, loader):
        self.model.eval()
        
        log_path = os.path.join(self.result_path, "selected_case.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)
        cam_weight = self.get_cam_weight()

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                if not os.path.exists(os.path.join(self.result_path, img_name, "feat_map")):
                    os.makedirs(os.path.join(self.result_path, img_name, "feat_map"))
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                feat_maps, confidence, _, rep = self.step(case_batch)
                input_image = case_batch[0].permute(1, 2, 0)
                final_map, final_seg = self.CAM_algo(input_image, feat_maps, cam_weight, img_name, print_feat_map=True)
                
                seg_gt = (seg_image*4).astype(np.uint8)

                whole_gt = np.where(seg_gt!=0, 1, 0)

                dice, iou, precision, recall = compute_seg_metrics(whole_gt, final_seg)
                
                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print("Dice Score:", f"{dice:.3f}")
                print("IOU Score: ", f"{iou:.3f}")
                print("Precision:", f"{precision:.3f}")
                print("Recall: ", f"{recall:.3f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: {dice:.3f}\n")
                log_file.writelines(f"IOU Score: {iou:.3f}\n")
                log_file.writelines(f"Precision: {precision:.3f}\n")
                log_file.writelines(f"Recall: {recall:.3f}\n")
                log_file.close()

                final_map = self.heatmap_postprocess(final_map)
                input_image, mix_image = self.img_fusion(input_image, final_map)

                io.imsave(os.path.join(self.result_path, img_name, f"input_{img_name}.jpg"), img_as_ubyte(input_image), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"whole_seg_{img_name}.jpg"), img_as_ubyte(whole_gt.astype(np.float32)), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"heat_{img_name}.jpg"), img_as_ubyte(final_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"mix_{img_name}.jpg"), img_as_ubyte(mix_image), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"final_seg_{img_name}.jpg"), img_as_ubyte(final_seg.astype(np.float32)), check_contrast=False)
                print_seg_contour(self.result_path, input_image, whole_gt.astype(np.float32), final_seg.astype(np.float32), img_name)


    def heatmap_postprocess(self, feat_map):
        heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        return heatmap

    def img_fusion(self, image, heatmap):
        cam = heatmap + np.float32(image)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return image, cam

    def run_tumor_test(self, loader):
        self.model.eval()
        
        
        result_metric = {'dice':[], 'iou':[], 'precision':[], 'recall':[]}
        log_path = os.path.join(self.result_path, "tumor_result.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)
        cam_weight = self.get_cam_weight()

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                feat_maps, confidence, _, rep = self.step(case_batch)
                
                input_image = case_batch[0].permute(1, 2, 0)
                final_map, final_seg = self.CAM_algo(input_image, feat_maps, cam_weight, img_name)
                
                seg_gt = (seg_image*4).astype(np.uint8)

                whole_gt = np.where(seg_gt!=0, 1, 0)
                
                dice, iou, precision, recall = compute_seg_metrics(whole_gt, final_seg)
                
                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print("Dice Score:", f"{dice:.3f}")
                print("IOU Score: ", f"{iou:.3f}")
                print("Precision:", f"{precision:.3f}")
                print("Recall: ", f"{recall:.3f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: {dice:.3f}\n")
                log_file.writelines(f"IOU Score: {iou:.3f}\n")
                log_file.writelines(f"Precision: {precision:.3f}\n")
                log_file.writelines(f"Recall: {recall:.3f}\n")
                log_file.close()

                result_metric["dice"].append(dice)
                result_metric["iou"].append(iou)
                result_metric["precision"].append(precision)
                result_metric["recall"].append(recall)

        return result_metric
    
    def run_normal_test(self, loader):
        self.model.eval()
        
        
        result_metric = {'dice':[], 'iou':[]}
        log_path = os.path.join(self.result_path, "normal_result.log")
        log_file = open(log_path, "a")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)
        cam_weight = self.get_cam_weight()

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                seg_image = seg_batch[0].permute(1, 2, 0).squeeze(2).numpy()
                feat_maps, confidence, _, rep = self.step(case_batch)

                input_image = case_batch[0].permute(1, 2, 0)
                final_map, final_seg = self.CAM_algo(input_image, feat_maps, cam_weight, img_name)
                
                seg_gt = (seg_image*4).astype(np.uint8)

                whole_gt = np.where(seg_gt!=0, 1, 0)
                dice, iou, _, _ = compute_seg_metrics(whole_gt, final_seg)
                
                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print("Dice Score:", f"{dice:.3f}")
                print("IOU Score: ", f"{iou:.3f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: {dice:.3f}\n")
                log_file.writelines(f"IOU Score: {iou:.3f}\n")
                log_file.close()

                result_metric["dice"].append(dice)
                result_metric["iou"].append(iou)

        return result_metric

    def CAM_algo(self, input_image, feat_maps, cam_weight, img_name, print_feat_map=False, output_hist=False):
        img_size = input_image.shape[0], input_image.shape[1]
        feat_maps = feat_maps.numpy()
        feat_maps = feat_maps.squeeze(0)
        feat_map = np.sum(feat_maps*cam_weight[:, np.newaxis, np.newaxis], axis=0)

        if (feat_map.max() - feat_map.min()) > 0:
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
        feat_map = torch.Tensor(feat_map)
        if print_feat_map:
            for idx, feat in enumerate(feat_map):
                heat_feat = self.heatmap_postprocess(feat)
                io.imsave(os.path.join(self.result_path, img_name, "feat_map", f"{idx}.jpg"), img_as_ubyte(heat_feat), check_contrast=False)
        final_map = F.interpolate(feat_map.unsqueeze(0).unsqueeze(1), size=img_size, mode='bilinear', align_corners=False)
        final_map = final_map.squeeze(1).squeeze(0).numpy()
        final_map = 1-final_map
        final_seg = gen_seg_mask(input_image, final_map, img_name, self.result_path, output_hist=output_hist)

        return final_map, final_seg