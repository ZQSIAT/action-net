import os
import shutil
import time
from multiprocessing import Pool


def aggregate_pku_label_list(label_dir):
    txt_list = os.listdir(label_dir)
    txt_list = list(filter(lambda x: x.endswith('.txt'), txt_list))
    txt_list.sort()

    label_sequence = []
    seqs_count = 0
    for txt_i in txt_list:
        txt_file = label_dir + "/" + txt_i
        with open(txt_file) as lf:
            for line in lf.readlines():
                label_line = line.strip("\n").split(",")

                label_line = [int(i) for i in label_line]

                # video_name, action_no, frame_start, frame_end
                assert label_line[2]>label_line[1], "{:s} Action {:d}: end frame should be large than start frame".format(txt_file,  label_line[0])

                label_sequence.append([txt_i.replace(".txt", ""),
                                       label_line[0], label_line[1], label_line[2]])

                seqs_count += 1
            lf.close()
    assert seqs_count == len(label_sequence)

    return label_sequence


def split_pku_action(label_sequence):
    video_name, action_str, frms_start, frms_end = label_sequence
    no = 0

    for fi in range(frms_start, frms_end):
        img_old_path = "{:s}/{:s}/depth-{:06d}.png".format(depth_dir, video_name, fi)
        img_new_dir = "{:s}/V{:s}_A{:02d}".format(depth_split_dir, video_name, action_str)

        if not os.path.isdir(img_new_dir):
            os.makedirs(img_new_dir)

        img_new_path = "{:s}/Depth-{:06d}.png".format(img_new_dir, no)

        shutil.copyfile(img_old_path, img_new_path)
        no += 1
    print("copy {:d} images to '{:s}'".format(no, img_new_dir))

    return no



if __name__ == "__main__":

    depth_dir = "/data/zqs/datasets/pku_mmd/depth"
    label_dir = "/data/zqs/datasets/pku_mmd/Label"
    depth_split_dir = "/data/zqs/datasets/pku_mmd/depth_split_xp"

    if not os.path.isdir(depth_split_dir):
        os.makedirs(depth_split_dir)

    png_format = "depth-{:06d}.png"

    label_sequence = aggregate_pku_label_list(label_dir)

    print(len(label_sequence))

    print("Start processing...")
    starttime = time.time()
    print("-" * 120)

    # processing by pool map
    p = Pool(32)


    res = p.map(split_pku_action, label_sequence)

    endtime = time.time()
    print("-" * 120)
    print("Finished! Time elapse: {:.2f} hours.".format((endtime - starttime)/3600.0))




