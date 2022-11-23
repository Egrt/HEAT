'''
Author: [egrt]
Date: 2022-08-23 11:44:15
LastEditors: Egrt
LastEditTime: 2022-11-23 15:25:35
Description: HEAT的模型加载与预测
'''
from turtle import pos
import torch
import torch.nn as nn
from models.resnet import ResNetBackbone
from models.corner_models import HeatCorner
from models.edge_models import HeatEdge
from models.corner_to_edge import get_infer_edge_pairs
from datasets.data_utils import get_pixel_features
from huggingface_hub import hf_hub_download
from PIL import Image
from utils import image_utils
from osgeo import gdal, ogr, osr
from tqdm import tqdm
import os
import scipy
import numpy as np
import cv2
import skimage

class HEAT(object):
    #-----------------------------------------#
    #   注意修改model_path
    #-----------------------------------------#
    _defaults = {
        #-----------------------------------------------#
        #  model_data指向整体网络的地址
        #-----------------------------------------------#
        "model_data"        : 'model_data/heat_checkpoints/checkpoints/ckpts_heat_outdoor_256/checkpoint.pth',
        #-----------------------------------------------#
        #   image_size模型预测图像的像素大小
        #-----------------------------------------------#
        "image_size"       : [256, 256], 
        #-----------------------------------------------#
        #   patch_size为模型切片的大小
        #-----------------------------------------------#
        "patch_size"        : 512,
        #-----------------------------------------------#
        #   patch_overlap为切片重叠像素
        #-----------------------------------------------#
        "patch_overlap"     : 0,
        #-----------------------------------------------#
        #   corner_thresh为预测角点的阈值大小
        #-----------------------------------------------#
        "corner_thresh"     : 0.01,    
        #-----------------------------------------------#
        #   基于角点候选数的最大边数（不能大于6）
        #-----------------------------------------------#
        "corner_to_edge_multiplier": 3,
        #-----------------------------------------------#
        #   边缘推理筛选的迭代次数
        #-----------------------------------------------#
        "infer_times"       : 3,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    #---------------------------------------------------#
    #   初始化MASKGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    def generate(self):
        # 从Huggingface加载整体网络模型
        filepath = hf_hub_download(repo_id="Egrt/HEAT", filename="checkpoint.pth")
        self.model = torch.load(filepath)
        # 加载Backbone
        self.backbone = ResNetBackbone()
        strides = self.backbone.strides
        num_channels = self.backbone.num_channels
        self.backbone = nn.DataParallel(self.backbone)
        self.backbone = self.backbone.cuda()
        self.backbone.eval()
        # 加载角点检测模型
        self.corner_model = HeatCorner(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                                backbone_num_channels=num_channels)
        self.corner_model = nn.DataParallel(self.corner_model)
        self.corner_model = self.corner_model.cuda()
        self.corner_model.eval()
        # 加载边缘检测模型
        self.edge_model = HeatEdge(input_dim=128, hidden_dim=256, num_feature_levels=4, backbone_strides=strides,
                            backbone_num_channels=num_channels)
        self.edge_model = nn.DataParallel(self.edge_model)
        self.edge_model = self.edge_model.cuda()
        self.edge_model.eval()
        # 分别加载模型的地址
        self.backbone.load_state_dict(self.model['backbone'])
        self.corner_model.load_state_dict(self.model['corner_model'])
        self.edge_model.load_state_dict(self.model['edge_model'])
            
    def detect_one_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        # 这里判断图片是否需要分成多个patch
        if image.size[0] < self.patch_size or image.size[1] < self.patch_size:
            is_slice = False
        else:
            is_slice = True
        if is_slice:
            # 复制原图
            image       = np.array(image, dtype=np.uint8)
            # 复制输入的原图
            viz_image   = image.copy()
            height, width = image.shape[0], image.shape[1]
            # 获取缩放比例
            scale = self.patch_size / self.image_size[0]
            # 初始化角点、边缘列表
            pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np = [], [], [], [], []
            # 开始切分
            stride = self.patch_size - self.patch_overlap
            patch_boundingboxes = image_utils.compute_patch_boundingboxes((height, width),
                                                                      stride=stride,
                                                                      patch_res=self.patch_size)
            edge_len = 0
            # 获取切分后的图片
            for bbox in tqdm(patch_boundingboxes, desc="使用切分进行预测", leave=False):
                # 切分图像
                crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                # np转Image类
                crop_image = Image.fromarray(crop_image)
                try:
                    pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np, _ = self.predict_no_patching(crop_image)
                except RuntimeError as e:
                    print("ERROR: " + str(e))
                    print("INFO: 减小patch_size 直到适合内存")
                    raise e
                # 拼接角点数组
                pred_corners[:, 0] = pred_corners[:, 0] * scale + bbox[0]
                pred_corners[:, 1] = pred_corners[:, 1] * scale + bbox[1]
                pred_corners_viz = pred_corners
                viz_image   = visualize_cond_generation(pred_corners_viz, pred_confs, viz_image, edges=pos_edges, 
                                edge_confs=edge_confs, shpfile=False)
            
            hr_image = Image.fromarray(np.uint8(viz_image))
        else:
            pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np, viz_image = self.predict_no_patching(image)
            #---------------------------------------------------------#
            #   此处推理结束
            #   开始在原图上根据角点坐标绘制角点与边缘
            #---------------------------------------------------------#
            pred_corners_viz = pred_corners
            image_result = visualize_cond_generation(pred_corners_viz, pred_confs, viz_image, edges=pos_edges, 
                            edge_confs=edge_confs, shpfile=True)
            hr_image = Image.fromarray(np.uint8(image_result))
        return hr_image
        
    #---------------------------------------------------------#
    #   不使用切片预测图像
    #   返回预测后的角点坐标、边缘
    #---------------------------------------------------------#      
    def predict_no_patching(self, image):
        image       = image.resize(tuple(self.image_size), Image.BICUBIC)
        # 将Image类转换为numpy
        image       = np.array(image, dtype=np.uint8)
        # 复制输入的原图
        viz_image   = image.copy()
        # preprocess image  numpy->tensor
        image       = process_image(image)
        #   获取所有像素的位置编码, 默认的图像尺度为256
        pixels, pixel_features = get_pixel_features(image_size=self.image_size[0])
        #   开始模型的预测
        with torch.no_grad():

            image_feats, feat_mask, all_image_feats = self.backbone(image)
            pixel_features = pixel_features.unsqueeze(0).repeat(image.shape[0], 1, 1, 1)
            preds_s1       = self.corner_model(image_feats, feat_mask, pixel_features, pixels, all_image_feats)

            c_outputs = preds_s1
            # 获取预测出的角点
            c_outputs_np = c_outputs[0].detach().cpu().numpy()
            # 筛选出大于阈值的角点的坐标
            pos_indices = np.where(c_outputs_np >= self.corner_thresh)
            pred_corners = pixels[pos_indices]
            # 获取对应预测角点的置信度
            pred_confs = c_outputs_np[pos_indices]
            # 根据预测角点的置信度进行非极大抑制
            pred_corners, pred_confs = corner_nms(pred_corners, pred_confs, image_size=c_outputs.shape[1])
            # 对角点两两排列组合，获取所有的角点对
            pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(pred_corners, pred_confs)
            # 获取角点数量
            corner_nums = torch.tensor([len(pred_corners)]).to(image.device)
            max_candidates = torch.stack([corner_nums.max() * self.corner_to_edge_multiplier] * len(corner_nums), dim=0)
            # 无序不重复集合
            all_pos_ids = set()
            # 边缘置信度字典
            all_edge_confs = dict()
            # 推理的迭代次数为3次
            for tt in range(self.infer_times):
                if tt == 0:
                    # gt_values和边缘掩膜大小一样且初始值为0
                    gt_values = torch.zeros_like(edge_mask).long()
                    # 第一二维度的数值设置为2
                    gt_values[:, :] = 2

                # 开始预测边缘
                s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = self.edge_model(image_feats, 
                    feat_mask,pixel_features,edge_coords, edge_mask,gt_values, corner_nums,max_candidates,True)
                num_total = s1_logits.shape[2]
                num_selected = selected_ids.shape[1]
                num_filtered = num_total - num_selected
                # 将输出值固定为(0,1)之间的概率分布
                s1_preds = s1_logits.squeeze().softmax(0)
                s2_preds_rel = s2_logits_rel.squeeze().softmax(0)
                s2_preds_hb = s2_logits_hb.squeeze().softmax(0)
                s1_preds_np = s1_preds[1, :].detach().cpu().numpy()
                s2_preds_rel_np = s2_preds_rel[1, :].detach().cpu().numpy()
                s2_preds_hb_np = s2_preds_hb[1, :].detach().cpu().numpy()

                selected_ids = selected_ids.squeeze().detach().cpu().numpy()
                # 进行筛选，将(0.9, 1)之间的设置为T，将(0.01,0.9)之间的设置为U,(0,0.01)之间的设置为F
                if tt != self.infer_times - 1:
                    s2_preds_np = s2_preds_hb_np

                    pos_edge_ids = np.where(s2_preds_np >= 0.9)
                    neg_edge_ids = np.where(s2_preds_np <= 0.01)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids[pos_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]
                        gt_values[0, actual_id] = 1
                    for neg_id in neg_edge_ids[0]:
                        actual_id = selected_ids[neg_id]
                        if gt_values[0, actual_id] != 2:
                            continue
                        gt_values[0, actual_id] = 0
                    num_to_pred = (gt_values == 2).sum()
                    if num_to_pred <= num_filtered:
                        break
                else:
                    s2_preds_np = s2_preds_hb_np

                    pos_edge_ids = np.where(s2_preds_np >= 0.5)
                    for pos_id in pos_edge_ids[0]:
                        actual_id = selected_ids[pos_id]
                        if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                            continue
                        all_pos_ids.add(actual_id)
                        all_edge_confs[actual_id] = s2_preds_np[pos_id]
            pos_edge_ids = list(all_pos_ids)
            edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
            pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
            edge_confs = np.array(edge_confs)

            if self.image_size[0] != 256:
                pred_corners = pred_corners / (self.image_size[0] / 256)

        return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np, viz_image
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
#---------------------------------------------------------#
#   根据角点的置信度排序，并筛选出大于置信度的角点坐标
#---------------------------------------------------------#
def corner_nms(preds, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(preds)):
        data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = scipy.ndimage.filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = scipy.ndimage.filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_preds, new_confs

def process_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = skimage.img_as_float(img)
    img = img.transpose((2, 0, 1))
    img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
    img = torch.Tensor(img).cuda()
    img = img.unsqueeze(0)
    return img

def postprocess_preds(corners, confs, edges):
    corner_degrees = dict()
    for edge_i, edge_pair in enumerate(edges):
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    else:
        good_corners = corners[good_ids]
        good_confs = confs[good_ids]
        id_mapping = {value: idx for idx, value in enumerate(good_ids)}
        new_edges = list()
        for edge_pair in edges:
            new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
            new_edges.append(new_pair)
        new_edges = np.array(new_edges)
        return good_corners, good_confs, new_edges

#---------------------------------------------------------#
#   将输入图像根据角点坐标进行可视化处理
#   不同于源代码，我们需要直接返回图像对象而不是保存到指定地址
#---------------------------------------------------------#
def visualize_cond_generation(positive_pixels, confs, image, gt_corners=None, prec=None, recall=None,
                              image_masks=None, edges=None, edge_confs=None, shpfile=False):
    # 复制原图  
    image = image.copy()
    if confs is not None:
        viz_confs = confs

    if edges is not None:
        preds = positive_pixels.astype(int)
        c_degrees = dict()
        for edge_i, edge_pair in enumerate(edges):
            conf = (edge_confs[edge_i] * 2) - 1
            cv2.line(image, tuple(preds[edge_pair[0]]), tuple(preds[edge_pair[1]]), (255 * conf, 255 * conf, 0), 2)
            c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
            c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1

    for idx, c in enumerate(positive_pixels):
        if edges is not None and idx not in c_degrees:
            continue
        if confs is None:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        else:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255 * viz_confs[idx]), -1)
        # if edges is not None:
        #    cv2.putText(image, '{}'.format(c_degrees[idx]), (int(c[0]), int(c[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
        #                0.5, (255, 0, 0), 1, cv2.LINE_AA)

    if gt_corners is not None:
        for c in gt_corners:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)

    if image_masks is not None:
        mask_ids = np.where(image_masks == 1)[0]
        for mask_id in mask_ids:
            y_idx = mask_id // 64
            x_idx = (mask_id - y_idx * 64)
            x_coord = x_idx * 4
            y_coord = y_idx * 4
            cv2.rectangle(image, (x_coord, y_coord), (x_coord + 3, y_coord + 3), (127, 127, 0), thickness=-1)

    # if confs is not None:
    #    cv2.putText(image, 'max conf: {:.3f}'.format(confs.max()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
    #                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if prec is not None:
        if isinstance(prec, tuple):
            cv2.putText(image, 'edge p={:.2f}, edge r={:.2f}'.format(prec[0], recall[0]), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'region p={:.2f}, region r={:.2f}'.format(prec[1], recall[1]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, 'prec={:.2f}, recall={:.2f}'.format(prec, recall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # 是否生成shp文件
    if shpfile:
        preds = positive_pixels.astype(int)
        # 获取点列表
        Polyline = []
        for edge_i, edge_pair in enumerate(edges):
            Polyline.append([preds[edge_pair[0]], preds[edge_pair[1]]])
        Polyline = np.array(Polyline, dtype=np.int32)
        # 写入shp文件
        writeShp(save_file_dir="shpfile", Polyline=Polyline)


    return image

def writeShp(save_file_dir="shpfile", Polyline=None):
    # 创建文件夹
    if os.path.exists(save_file_dir) is False:
        os.makedirs(save_file_dir)
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 属性表字段支持中文
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    # 注册驱动
    ogr.RegisterAll()
    # 创建shp数据
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        return "驱动不可用："+strDriverName
    # 创建数据源
    file_path = os.path.join(save_file_dir, "result.shp")
    oDS = oDriver.CreateDataSource(file_path)
    if oDS == None:
        return "创建文件失败：result.shp"
    if Polyline is not None:
        # 创建一个多边形图层，指定坐标系为WGS84
        papszLCO = []
        geosrs = osr.SpatialReference()
        geosrs.SetWellKnownGeogCS("WGS84")
        # 线：ogr_type = ogr.wkbLineString
        # 点：ogr_type = ogr.wkbPoint
        ogr_type = ogr.wkbMultiLineString
        # 面的类型为Polygon，线的类型为Polyline，点的类型为Point
        oLayer = oDS.CreateLayer("Polyline", geosrs, ogr_type, papszLCO)
        if oLayer == None:
            return "图层创建失败！"
        # 创建属性表
        # 创建id字段
        oId = ogr.FieldDefn("id", ogr.OFTInteger)
        oLayer.CreateField(oId, 1)
        # 创建name字段
        oName = ogr.FieldDefn("name", ogr.OFTString)
        oLayer.CreateField(oName, 1)
        oDefn = oLayer.GetLayerDefn()
        # 创建要素
        # 数据集
        # wkt_geom id name
        point_str_list = ['({} {},{} {})'.format(row[0, 0], row[0, 1], row[1, 0], row[1, 1]) for row in Polyline]
        Polyline_Wkt = ','.join(point_str_list)
        features = ['Polyline0;MULTILINESTRING({})'.format(Polyline_Wkt)]
        for index, f in enumerate(features):
            oFeaturePolygon = ogr.Feature(oDefn)
            oFeaturePolygon.SetField("id",index)
            oFeaturePolygon.SetField("name",f.split(";")[0])
            geomPolygon = ogr.CreateGeometryFromWkt(f.split(";")[1])
            oFeaturePolygon.SetGeometry(geomPolygon)
            oLayer.CreateFeature(oFeaturePolygon)
        # 创建完成后，关闭进程
        oDS.Destroy()
    return "数据集创建完成！"