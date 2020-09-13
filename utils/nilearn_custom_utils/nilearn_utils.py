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

    img = check_niimg(img)  # 传入的是单张图像。在utils/normalize.py的find_downsized_info函数中调用，将数据的前景并集mask作为实参传给img
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    # 不妨设infinity_norm = 1 (比如调用时传进来的mask就是一个0-1的前景mask)
    # passes_threshold可以视为一种mask, 数据data灰度值gray满足 gray < -(1e-8) or gray > 1e-8 的像素，在mask中为True
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
    if data.ndim == 4:  # data: ndarray, ndim是看data总共有几维，即data.ndim == len(data.shape)
        # 4D图像，沿着最后一维（各行横向遍历）检查是否有true. 这部分目的是检查每行是否有前景像素。
        passes_threshold = np.any(passes_threshold, axis=-1)    # check沿着指定维是否有true.
    coords = np.array(np.where(passes_threshold))   # 拿到前景像素的坐标/索引，是一个二维矩阵，coords[0], coords[1], coords[2]都是一个数组，数组内元素是axis=0, 1, 2的索引。
    # 下两行达到的目的：start和end都是三个值，索引从0到2分别对应具备非零像素的切片，行和列的最小值start和最大值end
    start = coords.min(axis=1)      # 对非零像素坐标，分别取各维的最小值。结果是一个包含3个值的vector, start[0]是非零像素索引中，axis=0的索引最小值。start[1], start[2]同理。
    end = coords.max(axis=1) + 1    # 对非零像素坐标，分别取各维的最大值。结果意义与上一行同理。

    # pad with one voxel to avoid resampling problems
    # 我们想保留图像非零区域，并额外保留一圈零像素。下面这两行是在进行越界检查。
    start = np.maximum(start - 1, 0)    # 标量0会广播地与start-1比较，取最大。例如np.maximum([1,2,3], 5) → array([5, 5, 5])。下同。
    end = np.minimum(end + 1, data.shape[:3])
    
    # 有点乱，举个例子：根据上面的代码及含义，start的三个元素为非零像素点在axis=0, 1, 2上的最小值，不妨设start=[0, 1, 0]
    # 同理，设end=[2, 2, 2]. 于是zip(start, end) = [(0,2), (1,2), (0,2)].
    slices = [slice(s, e) for s, e in zip(start, end)]  # slices = [slice(0,2), slice(1,2), slice(0,2)]. 注：slice(s, e)为切片对象，用其作索引相当于使用切片语法[s:e].
    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)  # 似乎是根据给定的切片列表slices, 对img进行切片？
