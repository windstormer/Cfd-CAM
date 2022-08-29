import torch
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

class ScoreCAM(object):
    def __init__(self, args, exp_name, encoder_model_type, gpuid):
        # encoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "encoder.pth")
        # decoder_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "decoder.pth")
        model_path = os.path.join(args.project_path, "record/CNet", args.pretrained_path, "model", "model.pth")

        self.model_model_type = encoder_model_type
        model = Res18_Classifier().cuda()
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

    def run_selected_case(self, loader):
        self.model.eval()
        
        
        log_path = os.path.join(self.result_path, "selected_case.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                if not os.path.exists(os.path.join(self.result_path, img_name, "feat_map")):
                    os.makedirs(os.path.join(self.result_path, img_name, "feat_map"))

                case_batch_1x = case_batch
                seg_batch_1x = seg_batch
                seg_image = seg_batch_1x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_1x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_1x)
                input_image_1x = case_batch_1x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_1x))

                final_map_1x, final_seg_1x = self.ScoreCAM_algo(input_image_1x, feat_maps, base_logit, img_name)

                dice_1x, iou_1x, precision_1x, recall_1x = compute_seg_metrics(whole_gt_1x, final_seg_1x)

                case_batch_2x = F.interpolate(case_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_batch_2x = F.interpolate(seg_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_image = seg_batch_2x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_2x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_2x)
                input_image_2x = case_batch_2x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_2x))

                final_map_2x, final_seg_2x = self.ScoreCAM_algo(input_image_2x, feat_maps, base_logit, img_name)

                dice_2x, iou_2x, precision_2x, recall_2x = compute_seg_metrics(whole_gt_2x, final_seg_2x)

                final_map_2x = torch.from_numpy(final_map_2x).unsqueeze(0).unsqueeze(1)
                final_seg_2x = torch.FloatTensor(final_seg_2x).unsqueeze(0).unsqueeze(1)
                final_map_2x = F.interpolate(final_map_2x, scale_factor=0.5)
                final_seg_2x = F.interpolate(final_seg_2x, scale_factor=0.5)
                final_map_2x = final_map_2x.squeeze(1).squeeze(0).numpy()
                final_seg_2x = final_seg_2x.squeeze(1).squeeze(0).numpy()
                final_seg_2x = np.round(final_seg_2x).astype(np.uint8)

                final_map = (final_map_1x + final_map_2x) / 2

                final_seg = gen_seg_mask(input_image_1x, final_map, img_name, self.result_path)

                final_map_1x = self.heatmap_postprocess(final_map_1x)
                mix_image_1x = self.img_fusion(input_image_1x, final_map_1x)
                final_map_2x = self.heatmap_postprocess(final_map_2x)
                mix_image_2x = self.img_fusion(input_image_1x, final_map_2x)
                final_map = self.heatmap_postprocess(final_map)
                mix_image_final = self.img_fusion(input_image_1x, final_map)

                dice_mix, iou_mix, precision_mix, recall_mix = compute_seg_metrics(whole_gt_1x, final_seg)

                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}")
                print(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}")
                print(f"Precision: 1x: {precision_1x:.4f}, 2x: {precision_2x:.4f}, mix: {precision_mix:.4f}")
                print(f"Recall: 1x: {recall_1x:.4f}, 2x: {recall_2x:.4f}, mix: {recall_mix:.4f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}\n")
                log_file.writelines(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}\n")
                log_file.writelines(f"Precision: 1x: {precision_1x:.4f}, 2x: {precision_2x:.4f}, mix: {precision_mix:.4f}\n")
                log_file.writelines(f"Recall: 1x: {recall_1x:.4f}, 2x: {recall_2x:.4f}, mix: {recall_mix:.4f}\n")
                log_file.close()


                io.imsave(os.path.join(self.result_path, img_name, f"input_{img_name}.jpg"), img_as_ubyte(input_image_1x), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"whole_seg_{img_name}.jpg"), img_as_ubyte(whole_gt_1x.astype(np.float32)), check_contrast=False)

                io.imsave(os.path.join(self.result_path, img_name, f"heat_1x_{img_name}.jpg"), img_as_ubyte(final_map_1x), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"mix_1x_{img_name}.jpg"), img_as_ubyte(mix_image_1x), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"final_seg_1x_{img_name}.jpg"), img_as_ubyte(final_seg_1x.astype(np.float32)), check_contrast=False)

                io.imsave(os.path.join(self.result_path, img_name, f"heat_2x_{img_name}.jpg"), img_as_ubyte(final_map_2x), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"mix_2x_{img_name}.jpg"), img_as_ubyte(mix_image_2x), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"final_seg_2x_{img_name}.jpg"), img_as_ubyte(final_seg_2x.astype(np.float32)), check_contrast=False)

                io.imsave(os.path.join(self.result_path, img_name, f"heat_mix_{img_name}.jpg"), img_as_ubyte(final_map), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"mix_final_{img_name}.jpg"), img_as_ubyte(mix_image_final), check_contrast=False)
                io.imsave(os.path.join(self.result_path, img_name, f"final_seg_{img_name}.jpg"), img_as_ubyte(final_seg.astype(np.float32)), check_contrast=False)
                
                print_seg_contour(self.result_path, input_image_1x, whole_gt_1x.astype(np.float32), final_seg_1x.astype(np.float32), final_seg_2x.astype(np.float32), img_name)

    def heatmap_postprocess(self, feat_map):
        heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        return heatmap

    def img_fusion(self, image, heatmap):
        cam = heatmap + np.float32(image)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def run_tumor_test(self, loader):
        self.model.eval()
        
        
        result_metric = {'dice':[], 'iou':[], 'precision':[], 'recall':[]}
        log_path = os.path.join(self.result_path, "tumor_result.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                case_batch_1x = case_batch
                seg_batch_1x = seg_batch
                seg_image = seg_batch_1x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_1x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_1x)
                
                input_image_1x = case_batch_1x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_1x))

                final_map_1x, final_seg_1x = self.ScoreCAM_algo(input_image_1x, feat_maps, base_logit, img_name)

                dice_1x, iou_1x, precision_1x, recall_1x = compute_seg_metrics(whole_gt_1x, final_seg_1x)

                case_batch_2x = F.interpolate(case_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_batch_2x = F.interpolate(seg_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_image = seg_batch_2x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_2x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_2x)
                input_image_2x = case_batch_2x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_2x))

                final_map_2x, final_seg_2x = self.ScoreCAM_algo(input_image_2x, feat_maps, base_logit, img_name)

                dice_2x, iou_2x, precision_2x, recall_2x = compute_seg_metrics(whole_gt_2x, final_seg_2x)
                
                final_map_2x = torch.from_numpy(final_map_2x).unsqueeze(0).unsqueeze(1)
                final_map_2x = F.interpolate(final_map_2x, scale_factor=0.5)
                final_map_2x = final_map_2x.squeeze(1).squeeze(0).numpy()

                final_map = (final_map_1x + final_map_2x) / 2

                final_seg = gen_seg_mask(input_image_1x, final_map, img_name, self.result_path)

                dice_mix, iou_mix, precision_mix, recall_mix = compute_seg_metrics(whole_gt_1x, final_seg)

                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}")
                print(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}")
                print(f"Precision: 1x: {precision_1x:.4f}, 2x: {precision_2x:.4f}, mix: {precision_mix:.4f}")
                print(f"Recall: 1x: {recall_1x:.4f}, 2x: {recall_2x:.4f}, mix: {recall_mix:.4f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}\n")
                log_file.writelines(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}\n")
                log_file.writelines(f"Precision: 1x: {precision_1x:.4f}, 2x: {precision_2x:.4f}, mix: {precision_mix:.4f}\n")
                log_file.writelines(f"Recall: 1x: {recall_1x:.4f}, 2x: {recall_2x:.4f}, mix: {recall_mix:.4f}\n")
                log_file.close()
                
                result_metric["dice"].append(dice_mix)
                result_metric["iou"].append(iou_mix)
                result_metric["precision"].append(precision_mix)
                result_metric["recall"].append(recall_mix)

        return result_metric
    
    def run_normal_test(self, loader):
        self.model.eval()
        
        
        result_metric = {'dice':[], 'iou':[]}
        log_path = os.path.join(self.result_path, "normal_result.log")
        log_file = open(log_path, "a")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]
                case_batch_1x = case_batch
                seg_batch_1x = seg_batch
                seg_image = seg_batch_1x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_1x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_1x)
                input_image_1x = case_batch_1x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_1x))

                final_map_1x, final_seg_1x = self.ScoreCAM_algo(input_image_1x, feat_maps, base_logit, img_name)
                dice_1x, iou_1x, precision_1x, recall_1x = compute_seg_metrics(whole_gt_1x, final_seg_1x)

                case_batch_2x = F.interpolate(case_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_batch_2x = F.interpolate(seg_batch, scale_factor=2.0, mode='bilinear', align_corners=False)
                seg_image = seg_batch_2x[0].permute(1, 2, 0).squeeze(2).numpy()
                seg_gt = (seg_image*4).astype(np.uint8)
                whole_gt_2x = np.where(seg_gt!=0, 1, 0)
                feat_maps, confidence, _, rep = self.step(case_batch_2x)
                input_image_2x = case_batch_2x[0].permute(1, 2, 0)
                _, _, base_logit, _ = self.step(torch.zeros_like(case_batch_2x))

                final_map_2x, final_seg_2x = self.ScoreCAM_algo(input_image_2x, feat_maps, base_logit, img_name)
                dice_2x, iou_2x, precision_2x, recall_2x = compute_seg_metrics(whole_gt_2x, final_seg_2x)


                final_map_2x = torch.from_numpy(final_map_2x).unsqueeze(0).unsqueeze(1)
                final_map_2x = F.interpolate(final_map_2x, scale_factor=0.5)
                final_map_2x = final_map_2x.squeeze(1).squeeze(0).numpy()

                final_map = (final_map_1x + final_map_2x) / 2

                final_seg = gen_seg_mask(input_image_1x, final_map, img_name, self.result_path)

                dice_mix, iou_mix, _, _ = compute_seg_metrics(whole_gt_1x, final_seg)

                print("Img Name:", img_name, ", Confidence:", confidence[0].numpy())
                print(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}")
                print(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}")
                log_file = open(log_path, "a")
                log_file.writelines(f"Img Name: {img_name}, Confidence: {confidence[0].numpy()} \n")
                log_file.writelines(f"Dice Score: 1x: {dice_1x:.4f}, 2x: {dice_2x:.4f}, mix: {dice_mix:.4f}\n")
                log_file.writelines(f"IOU Score: 1x: {iou_1x:.4f}, 2x: {iou_2x:.4f}, mix: {iou_mix:.4f}\n")
                log_file.close()
                
                result_metric["dice"].append(dice_mix)
                result_metric["iou"].append(iou_mix)

        return result_metric

    def ScoreCAM_algo(self, input_image, feat_maps, base_logit, img_name, print_feat_map=False, output_hist=False):
        img_size = input_image.shape[0], input_image.shape[1]
        featured_tensor = []
        norm_feat_map = []
        for idx, feat_map in enumerate(feat_maps[0]):
            feat_map = F.interpolate(feat_map.unsqueeze(0).unsqueeze(1), size=img_size, mode='bilinear', align_corners=False)
            if (feat_map.max() - feat_map.min()) > 0:
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            feat_map = feat_map.permute(0, 2, 3, 1).squeeze(0)
            f = feat_map.repeat(1, 1, 3)
            featured_img = input_image * f
            masked_input = featured_img.unsqueeze(0).permute(0, 3, 1, 2)
            norm_feat_map.append(feat_map.squeeze(2).numpy())
            featured_tensor.append(masked_input)

        featured_tensor = torch.cat(featured_tensor)
        
        _, _, featured_logit, _ = self.step(featured_tensor.cuda())
        featured_logit = torch.flatten(featured_logit)
        featured_logit = featured_logit-torch.ones_like(featured_logit) * base_logit[0]
        
        if print_feat_map:
            for idx, feat in enumerate(norm_feat_map):
                heat_feat = self.heatmap_postprocess(feat)
                io.imsave(os.path.join(self.result_path, img_name, "feat_map", f"{idx}.jpg"), img_as_ubyte(heat_feat), check_contrast=False)
        
        norm_feat_map = np.asarray(norm_feat_map)

        cam_weight = F.softmax(featured_logit, dim=0).numpy()

        final_map = np.sum(norm_feat_map*cam_weight[:, np.newaxis, np.newaxis], axis=0)
        final_map = np.maximum(final_map, 0)
        final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min())

        final_seg = gen_seg_mask(input_image, final_map, img_name, self.result_path, output_hist=output_hist)

        return final_map, final_seg