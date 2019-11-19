import collections
import numpy as np
import math

class DepthSeqsGradient(object):
    def __init__(self, is_gradient, is_identity=False):
        self.is_gradient = is_gradient
        self.is_identity = is_identity

    def __call__(self, depth_seqs):
        """
        :param depth_seqs: shape[(C x F x H x W)]
        :return:
        """
        if depth_seqs.dtype != np.float:
            depth_seqs = depth_seqs.astype(np.float)

        if self.is_gradient:
            depth_seqs = depth_seqs[0, :, :, :]

            data_gradient = np.gradient(depth_seqs)

            [dt, dy, dx] = data_gradient
            d_norm = np.sqrt(np.square(dx) + np.square(dy) + np.square(dt) + 1)

            data_processed = np.stack([di / d_norm for di in data_gradient], axis=0)

            if self.is_identity:
                data_processed = np.concatenate([data_processed, d_norm[np.newaxis,:]], axis=0)


            #data_gradient = np.concatenate((data_gradient, np.ones(depth_seqs.shape)[np.newaxis,:]), axis=0)
            #data_gradient_norm = 1 / (np.sqrt(np.square(data_gradient).sum(axis=0, keepdims=True)))
            # old method
            # data_gradient = np.gradient(depth_seqs)
            # data_gradient_norm = 1 / (np.sqrt(np.square(data_gradient).sum(axis=0, keepdims=True)+1))
            # data_processed = data_gradient * data_gradient_norm.repeat(3, axis=0)
        else:
            data_processed = depth_seqs
        return data_processed


def sparse_sampling_frames_from_segments(length_original, segments, sampling_type="random"):
    """
    sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    if length_original > segments:

        segments_list = np.array_split(np.arange(0, length_original), segments)
        if sampling_type == "order":
            frame_idx = [si[0] for si in segments_list]
        else:
            frame_idx = [np.random.choice(si, 1)[0] for si in segments_list]

    else:

        frame_idx = list(range(length_original))

        for index in range(segments - length_original):
            if len(frame_idx) >= segments:
                break
            frame_idx.append(index)

    return frame_idx


def sparse_sampling_frames_from_segments_dual(length_original, segments, sampling_type="random"):
    """
    sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    # length_original = length_original - 1

    frms_list = list(range(0, length_original))

    if length_original < segments:
        tile_time = segments // length_original + 1
        frms_list = np.tile(frms_list, tile_time)
    # else:
    #     frms_list = list(range(0, length_original))

    if sampling_type == "duration_first":
        frame_idx = list(range(0, segments))
    else:
        segments_list = np.array_split(frms_list, segments)
        if sampling_type == "fixed_middle":
            frame_idx = []
            for si in segments_list:
                sidx = len(si) // 2
                # frame_idx.append(si[sidx])
                frame_idx.append(si[sidx] + 1)
                # frame_idx.append(si[sidx] + 2)
        else:
            frame_idx = []
            for si in segments_list:
                pre_idx = np.random.choice(si, 1)[0]
                # frame_idx.append(pre_idx)
                frame_idx.append(pre_idx + 1)
                # frame_idx.append(pre_idx + 2)

    frame_idx = [length_original-2 if fi >= length_original-1 else fi for fi in frame_idx]

    return frame_idx


def variant_sparse_sampling_frames_from_segments_dual(length_original, segments, sampling_type="random"):
    """
    sampling_frames_from_segments
    :param length_original:
    :param segments:
    :return:
    """
    # length_original = length_original - 1

    frms_list = list(range(0, length_original))

    if length_original < segments:
        tile_time = segments // length_original + 1
        frms_list = np.tile(frms_list, tile_time)
    # else:
    #     frms_list = list(range(0, length_original))

    if sampling_type == "duration_first":
        frame_idx = list(range(0, segments))
    else:
        segments_list = np.array_split(frms_list, segments)
        if sampling_type == "fixed_middle":
            frame_idx = []
            for si in segments_list:
                sidx = len(si) // 2
                frame_idx.append(si[sidx])
                frame_idx.append(si[sidx] + 1)
        else:
            frame_idx = []
            for si in segments_list:
                pre_idx = np.random.choice(si, 1)[0]
                frame_idx.append(pre_idx)
                frame_idx.append(pre_idx + 1)

    frame_idx = [length_original-1 if fi > length_original-1 else fi for fi in frame_idx]

    return frame_idx


if __name__ == '__main__':
    depth_seqs = np.random.randint(0, 4095, size=[4, 30, 240, 320])

    # expected_len = 20
    # stride = 2

    """
    temporal_transform = TemporalCompose([DepthSeqsCoarseCrop(expected_len, stride, "linear"),
                                          DepthSeqsCoarsePadding(expected_len, "last")])
    depth_seqs = temporal_transform.__call__(depth_seqs)

    print(depth_seqs.shape)
    """

    # length = depth_seqs.shape[1]

    # segments = 7
    #
    # pooling = DepthSeqsGradient(segments)
    #
    # depth_seqs = pooling(depth_seqs)
    #
    # print(depth_seqs.shape)
    # segments = 7
    # sampling_type = "average"
    # sampling = DepthSeqsSnapshotPooling(segments, sampling_type, "random")
    #
    # depth_seqs = sampling(depth_seqs)
    #
    # print(depth_seqs.shape)
    a = sparse_sampling_frames_from_segments_dual(110, 8, "fixed_middle")
    print(a)
    #
    # a = sparse_sampling_frames_from_segments(3, 7)
    # #np.sort(np.random.choice(np.arange(10), 7, replace=False))
    # print(a)





