import numpy as np
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to


def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    # 不妨设infinity_norm = 255
    # passes_threshold可以视为一种mask, 数据data灰度值满足gray < -(1e-8 * 255) 或 gray > 1e-8 * 255的像素，在mask中为True
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
    if data.ndim == 4:  # data: ndarray, ndim是看data总共有几维，即data.ndim == len(data.shape)
        # 4D图像，沿着最后一维（各行横向遍历）检查是否有true. 这部分目的是检查每行是否有前景像素。
        passes_threshold = np.any(passes_threshold, axis=-1)    # check沿着指定维是否有true.
    coords = np.array(np.where(passes_threshold))   # 拿到前景像素的坐标/索引
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]
    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)
