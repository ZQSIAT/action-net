import os
import cv2
import numpy as np
import logging
import numba

from datasets.UOWCombined3D import read_uow_depth_maps


@numba.jit(nogil=True)
def save_png_from_depth_maps(depth_maps, save_dir, bin_file_name):
    img_save_dir = "{:s}/{:s}".format(save_dir, bin_file_name.replace(".bin", ""))

    # create the directory if not exist
    if not os.path.isdir(img_save_dir):
        os.makedirs(img_save_dir)

    frms = depth_maps.shape[0]
    # from matplotlib import pyplot as plt

    for fi in range(frms):
        frame_name = "{img_dir:s}/Depth-{img_id:06d}.png".format(img_dir=img_save_dir,
                                                                 img_id=fi + 1)
        depth_data = depth_maps[fi]

        # plt.imshow(depth_data)
        # plt.colorbar()
        # plt.show()
        #
        depth_data = cv2.resize(depth_data, (320, 240), interpolation=cv2.INTER_NEAREST)

        # plt.imshow(depth_data)
        # plt.colorbar()
        # plt.show()
        # raise RuntimeError

        cv2.imwrite(frame_name, depth_data)


if __name__ == "__main__":
    from configs.param_config import ConfigClass
    config = ConfigClass()

    config.set_environ()

    # create logger
    logger_name = "img_saver"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create file handler
    log_path = "/data/xp_ji/datasets/UOW_Combined3D/uow_split_re_jit_320.log"
    file_handle = logging.FileHandler(log_path)
    console_handle = logging.StreamHandler()

    logger.addHandler(file_handle)
    logger.addHandler(console_handle)

    bin_dir = "/data/xp_ji/datasets/UOW_Combined3D/Depth/"
    depth_dir = "/data/xp_ji/datasets/UOW_Combined3D/Depth_img_320"

    bin_list = os.listdir(bin_dir)
    bin_list = list(filter(lambda x: x.endswith('bin'), bin_list))

    bin_list.sort()

    # cat all labels from label_dir
    seqs_count = len(bin_list)


    for i, bin_i in enumerate(bin_list):

        bin_file = bin_dir + "/" + bin_i
        # assert os.path.isfile(label_file), "Error: {:s} do not exist!".format(label_file)

        depth_maps = read_uow_depth_maps(bin_file) # FxHxW format

        depth_maps = depth_maps.astype(np.uint16)

        logger.info("{:.2f}% Processing {:s} "
                    .format(100.0 * float(i) / seqs_count,
                            bin_i))

        # spilt video by label file
        save_png_from_depth_maps(depth_maps, depth_dir, bin_i)

    logger.info("-"*120)
    logger.info("Finished! ")



