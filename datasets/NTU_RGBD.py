# -*- coding: utf-8 -*-
"""
This python file is designed to read the NTU RGB+D dataset
"""
from multiprocessing import Process
# import cv2
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import random
import math
from transforms.temporal_transforms import sparse_sampling_frames_from_segments_dual, variant_sparse_sampling_frames_from_segments_dual
import matplotlib
matplotlib.use("TkAgg")

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def get_path_list_form_file_list(file_format, file_list_file):
    """
    get_path_list_form_file_list:
    :param file_format:
    :param file_list_file:
    :return:
    """

    file_dir_list = []
    label_list = []
    frame_list = []

    with open(file_list_file, "r") as flp:
        for line in flp.readlines():
            flp_line = line.strip("\n").split("\t")

            # default format is "<$file_format>/t<$label>/t<[frames,height,width]>"
            target_file_path = file_format + "/" + flp_line[0]
            target_label = int(flp_line[1])
            target_length = int(flp_line[2])

            file_dir_list.append(target_file_path)
            label_list.append(target_label)
            frame_list.append(target_length)

        flp.close()

    # file_path = list(map(lambda x: file_format.format(x), file_list))
    # print(file_dir_list)
    # print(len(file_dir_list))
    # raise RuntimeError
    return file_dir_list, label_list, frame_list


class NTURGBD(data.Dataset):
    def __init__(self, index_param, spatial_transform=None, temporal_transform=None):
        self.modality = index_param["modality"]
        self.is_segmented = index_param["is_segmented"]
        self.is_depth_clip = index_param["is_depth_clip"]
        self.subset_type = index_param["subset_type"]
        self.temporal_param = index_param["temporal_param"]
        self.spatial_param = index_param["spatial_param"]
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        if self.is_segmented:
            self.depth_format = "MDepth-{idx:08d}.png"
            self.file_format = index_param["file_format"]#  + "_masked"
        else:
            self.depth_format = "Depth-{idx:08d}.png"
            self.file_format = index_param["file_format"]

        if self.is_depth_clip:
            self.file_format = self.file_format + "_clip"

        logger = index_param["logger"]
        # print(index_param["file_subset_list"])
        # raise RuntimeError
        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(self.file_format, index_param["file_subset_list"])

    def __getitem__(self, index):
        target_file_dir = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]

        target_data = self._loading_data(target_file_dir, target_length)
        # print(target_data)
        # raise RuntimeError

        return target_data, target_label, target_length

    def __len__(self):
        return len(self.file_list)

    def _loading_data(self, file_path, file_count):
        """
        _loading_transformed_data:
        :param file_path:
        :return: data_processed:
        """
        if self.modality == "Depth":

            temporal_param = self.temporal_param

            if temporal_param is not None:
                # using file idx to reduce the loading time
                if "dynamic_snapshot_sampling" in temporal_param.keys():
                    scales = temporal_param["dynamic_snapshot_sampling"]["scales"]
                    segments = temporal_param["dynamic_snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["dynamic_snapshot_sampling"]["sampling_type"]

                    file_count_scales = math.ceil(random.choice(scales) * file_count)

                    file_idx = sparse_sampling_frames_from_segments_dual(file_count_scales,
                                                                          segments,
                                                                          sampling_type)

                elif "variant_snapshot_sampling" in temporal_param.keys():
                    raise NotImplementedError
                    # segments = temporal_param["adjoin_snapshot_sampling"]["segments"]
                    # sampling_type = temporal_param["adjoin_snapshot_sampling"]["sampling_type"]
                    # file_idx = variant_sparse_sampling_frames_from_segments_dual(file_count,
                    #                                                              segments,
                    #                                                              sampling_type)

                elif "snapshot_sampling" in temporal_param.keys():
                    segments = temporal_param["snapshot_sampling"]["segments"]
                    sampling_type = temporal_param["snapshot_sampling"]["sampling_type"]
                    file_idx = sparse_sampling_frames_from_segments_dual(file_count,
                                                                          segments,
                                                                          sampling_type)
                else:
                    file_idx = None
            else:
                file_idx = None

            norm_val = False
            depth_clip = self.spatial_param["depth_clip"]
            region_crop = self.spatial_param["region_crop"]

            if self.is_depth_clip:
                data_processed = read_ntu_depth_maps_pil(file_path,
                                                         self.depth_format,
                                                         img_loader=pil_loader,
                                                         file_idx=file_idx,
                                                         norm_val=norm_val,
                                                         depth_clip=depth_clip,
                                                         region_crop=region_crop)

            else:
                data_processed = read_ntu_depth_maps(file_path,
                                                     self.depth_format,
                                                     img_loader=cv2_loader,
                                                     file_idx=file_idx,
                                                     norm_val=norm_val,
                                                     depth_clip=depth_clip,
                                                     region_crop=region_crop)

            if self.temporal_transform is not None:
                data_processed = self.temporal_transform(data_processed)
                # print(data_processed)
                # raise RuntimeError
            if self.spatial_transform is not None:
                data_processed = self.spatial_transform(data_processed)

        else:
            raise ValueError("Unknown modality: '{:s}'".format(self.modality))

        return data_processed


def read_ntu_depth_maps(depth_file_dir,
                        depth_format,
                        img_loader=None,
                        file_idx=None,
                        norm_val=False,
                        depth_clip=None,
                        region_crop=None):
    """

    :param depth_file:
    :param img_loader:
    :param size:
    :return: CxFxHxW
    """

    if file_idx is not None:
        depth_file_list = [depth_format.format(idx=idx+1) for idx in file_idx]
    else:
        depth_file_list = sorted(os.listdir(depth_file_dir))

    img_array_list = []

    for img_i in depth_file_list:
        img_path = depth_file_dir + "/" + img_i
        assert os.path.isfile(img_path), "{:s} does not exist!".format(img_path)

        img = img_loader(img_path)
        img_array_list.append(img)

    img_array = np.concatenate([np.expand_dims(x, 0) for x in img_array_list], axis=0)

    # img_array_crop = img_array[:, 90: 370, 110: 390]
    # img_array_crop = img_array[:, 90: 410, 90: 410]

    # img_array_crop = img_array_crop.astype(np.float)

    if region_crop is not None:
        h_s, h_e, w_s, w_e = region_crop
        img_array_crop = img_array[:, h_s: h_e, w_s: w_e]

    else:
        img_array_crop = img_array

    if depth_clip is not None:
        depth_min, depth_max = depth_clip
        img_array_crop = img_array_crop.astype(np.float)  # uint16 -> float
        img_array_crop = (img_array_crop - depth_min) / (depth_max-depth_min)  # valid 500-4500mm
        img_array_crop = np.clip(img_array_crop, 0, 1)
        img_array_crop = img_array_crop * 255.0

        # img_array_crop[img_array_crop < depth_min] = 0
        # img_array_crop[img_array_crop > depth_max] = 0

    if norm_val:
        # raise RuntimeError
        # img_array_crop = (img_array_crop - 500.0) / 4000.0  # valid 500-4500mm
        img_array_crop = img_array_crop / 8000.0
        depth_map = np.clip(img_array_crop, 0.0, 1.0)
        img_array_crop = depth_map * 255.0

    img_array_crop = img_array_crop[:, np.newaxis] # FxCxHxW


    return img_array_crop


def read_ntu_depth_maps_pil(depth_file_dir,
                            depth_format,
                            img_loader=None,
                            file_idx=None,
                            norm_val=False,
                            depth_clip=None,
                            region_crop=None):
    """

    :param depth_file:
    :param img_loader:
    :param size:
    :return:  pil list
    """

    if file_idx is not None:
        depth_file_list = [depth_format.format(idx=idx+1) for idx in file_idx]
    else:
        depth_file_list = sorted(os.listdir(depth_file_dir))

    img_pil_list = []

    for img_i in depth_file_list:
        img_path = depth_file_dir + "/" + img_i
        assert os.path.isfile(img_path), "{:s} does not exist!".format(img_path)

        img = img_loader(img_path)

        if region_crop is not None:
            h_s, h_e, w_s, w_e = region_crop
            img_pil_list.append(img.crop([h_s, w_s, h_e, w_e]))
        else:
            img_pil_list.append(img)

    assert img_pil_list[0].mode == 'L', "image mode is error"

    return img_pil_list


if __name__ == '__main__':
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    from matplotlib import pyplot as plt
    from transforms.group_transforms import group_data_transforms

    is_gradient = False
    is_depth_clip = False

    index_param = {}
    index_param["modality"] = "Depth"
    index_param["is_gradient"] = is_gradient
    index_param["is_segmented"] = True
    index_param["is_depth_clip"] = is_depth_clip

    index_param["file_format"] = "/ssd_data/zqs/datasets/ntu_rgb+d/depth_masked"
    index_param["file_subset_list"] = "/home/zqs/workspace-xp/deployment/action-net-47/NTU_RGB+D_output/protocols/TEST.txt"

    index_param["subset_type"] = "train"
    index_param["logger"] = None

    modality = "Depth"
    spatial_method = "spatial_group_crop_org"
    temporal_method = "dynamic_snapshot_sampling"

    loader_param = config.get_loader_param("NTU_RGB+D", modality)

    spatial_param = loader_param["spatial_transform"][spatial_method]["train"]
    temporal_param = loader_param["temporal_transform"][temporal_method]["train"]

    spatial_transform, temporal_transform = group_data_transforms(spatial_param,
                                                                  temporal_param,
                                                                  modality,
                                                                  is_gradient=is_gradient,
                                                                  is_depth_clip=is_depth_clip)

    # spatial_transform = None
    index_param["temporal_param"] = temporal_param
    index_param["spatial_param"] = spatial_param

    # print(index_param)
    # raise RuntimeError
    data_loader = NTURGBD(index_param,
                          spatial_transform=spatial_transform,
                          temporal_transform=temporal_transform)


    #print("length: {:d}".format(data_loader.__len__()))

    data_len = data_loader.__len__()
    print(data_len)

    depth_maps, label, length = data_loader.__getitem__(0)

    print(depth_maps.shape)

    if depth_maps.shape[0] == 3:
        depth_map = depth_maps[:, 1, :, :].squeeze(1).permute(1, 2, 0) + 1
    else:
        depth_map = depth_maps[0, 0, :, :]

    print(depth_map.shape)

    plt.imshow(depth_map)
    plt.colorbar()
    plt.show()








