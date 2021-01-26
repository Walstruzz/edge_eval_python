import os
import numpy as np
from scipy.io import loadmat
from .impl.toolbox import conv_tri, grad2
from .impl.nms import fast_edge_nms
import cv2


def nms_process_one_image(image, save_path=None, save=True):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    edge = fast_edge_nms(edge, ori, 1, 5, 1.01)
    edge = np.round(edge * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(model_name_list, result_dir, save_dir, key=None, file_format=".mat"):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)

    for model_name in model_name_list:
        model_save_dir = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for file in os.listdir(result_dir):
            if os.path.splitext(file)[-1] != file_format:
                continue
            abs_path = os.path.join(result_dir, file)
            if file_format == ".mat":
                assert key is not None
                image = loadmat(abs_path)[key]
            elif file_format == ".npy":
                image = np.load(abs_path)
            else:
                raise NotImplementedError
            save_name = os.path.join(model_save_dir, "{}.png".format(os.path.splitext(file)[0]))
            nms_process_one_image(image, save_name, True)


if __name__ == '__main__':
    nms_process("hed", "hed_result", "NMS_RESULT_FOLDER", key="result")
