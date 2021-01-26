from argparse import ArgumentParser

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
    parser = ArgumentParser("edge eval")
    parser.add_argument("--alg", type=str, default="HED", help="algorithms for plotting.")
    parser.add_argument("--model_name_list", type=str, default="hed", help="model name")
    parser.add_argument("--result_dir", type=str, default="examples/hed_result", help="results directory")
    parser.add_argument("--save_dir", type=str, default="examples/nms_result", help="nms result directory")
    parser.add_argument("--gt_dir", type=str, default="examples/bsds500_gt", help="ground truth directory")
    parser.add_argument("--key", type=str, default="result", help="key")
    parser.add_argument("--file_format", type=str, default=".mat", help=".mat or .npy")
    parser.add_argument("--workers", type=int, default="-1", help="number workers, -1 for all workers")
    args = parser.parse_args()


