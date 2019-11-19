
import torch
from schemes.model_metrics import calculate_accuracy_percent


eval_root = "/home/user007/workspace/deployment/action-net/NTU_RGB+D_output/evaluate_root"

if __name__ == "__main__":

    # D1_file_dir = eval_root + "/" + "1_cross_subjects_83.76_EVAL_P3D_34_D1_CS_S8_B64_S224_Resnet_P3D_34"
    # G3_file_dir = eval_root + "/" + "1_cross_subjects_88.58_EVAL_P3D_34_G3_CS_S8_B64_S224_Resnet_P3D_34"

    D1_file_dir = eval_root + "/" + "2_cross_view_84.41_EVAL_P3D_34_D1_CV_S8_B64_S224_Resnet_P3D_34"
    G3_file_dir = eval_root + "/" + "2_cross_view_91.51_EVAL_P3D_34_G3_CV_S8_B64_S224_Resnet_P3D_34"

    D1_file = "{:s}/Evaluate_result.pth.tar".format(D1_file_dir)
    G3_file = "{:s}/Evaluate_result.pth.tar".format(G3_file_dir)

    D1_sorce = torch.load(D1_file)["output_score"]
    G3_sorce = torch.load(G3_file)["output_score"]

    true_label_list = torch.load(D1_file)["true_label"]

    true_label = torch.LongTensor(true_label_list).unsqueeze(1).cuda()

    fusion_score = 1.0*D1_sorce + 3.0*G3_sorce

    print(true_label.shape)
    print(fusion_score.shape)

    # _, pred = fusion_score.topk(1, 1, True)
    #
    # print(pred.shape)


    predicted_accuracy, n_correct_elems = calculate_accuracy_percent(fusion_score, true_label)

    print("-"*60)
    print(predicted_accuracy)
    print(n_correct_elems)

