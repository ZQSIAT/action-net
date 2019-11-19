import os
import cv2
import time
import shutil
import numpy as np
from multiprocessing import Pool


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def get_file_list(rootDir):
    file_dir_list = []
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            if "png" in file and ".d" not in file:#[0] != ".":
                file_name = os.path.join(root, file)
                file_dir_list.append(file_name)
    return file_dir_list


def depth_value_clip(img_path):
    img_array = cv2_loader(img_path)

    depth_min, depth_max = depth_clip
    img_array = img_array.astype(np.float)  # uint16 -> float
    img_array_clip = (img_array - depth_min) / (depth_max - depth_min)  # valid 500-4500mm
    img_array_clip = np.clip(img_array_clip, 0, 1)
    img_array_clip = img_array_clip * 255.0
    img_array_clip = img_array_clip.astype(np.uint8)

    img_new_path = img_path.replace(depth_split_dir, data_new_dir)

    img_new_dir = data_new_dir + "/" + img_path.replace(depth_split_dir,"").split("/")[1]

    if not os.path.isdir(img_new_dir):
        os.makedirs(img_new_dir)
        print("creating directory: {:s}".format(img_new_dir))

    res = cv2.imwrite(img_new_path, img_array_clip, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    return res



if __name__ == "__main__":

    depth_split_dir = "/data5/zqs/datasets/pku_mmd/depth_split_xp"
    data_new_dir = "/data4/zqs/datasets/pku_mmd/depth_clipped"

    depth_clip = [500, 4500]

    if not os.path.isdir(data_new_dir):
        os.makedirs(data_new_dir)

    png_file_list = get_file_list(depth_split_dir)

    print("png file count: {:d}".format(len(png_file_list)))

    starttime = time.time()
    print("Start depth clip...")

    # print(png_file_list)

    multi_processing_pool = Pool(28)
    res = multi_processing_pool.map(depth_value_clip, png_file_list)
    # for i in png_file_list:
    #     multi_processing_pool.apply_async(depth_value_clip, [i])
    #
    # multi_processing_pool.join()
    # multi_processing_pool.close()

    # for i in range(0,100):
    #     res = depth_value_clip(png_file_list[i])

    print("-" * 120)
    # print("handle count:{:d}  list length: {:d}".format(len(res), len(png_file_list)))

    endtime = time.time()

    print("Finished! Time elapse: {:.2f} hours.".format((endtime - starttime) / 3600.0))


