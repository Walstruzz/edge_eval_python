from nms_process import nms_process
from eval_edge import eval_edge


def main():
    alg = ["HED"]  # algorithms for plotting
    model_name_list = ["hed"]  # model name
    result_dir = "examples/hed_result"  # forward result directory
    save_dir = "examples/nms_result"  # nms result directory
    gt_dir = "examples/bsds500_gt"  # ground truth directory
    key = "result"  # x = scipy.io.loadmat(filename)[key]
    file_format = ".mat"  # ".mat" or ".npy"
    workers = 16  # number workers
    nms_process(model_name_list, result_dir, save_dir, key, file_format)
    eval_edge(alg, model_name_list, result_dir, gt_dir, workers)


if __name__ == '__main__':
    main()

