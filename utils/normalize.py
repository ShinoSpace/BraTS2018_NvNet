import os

import numpy as np
from nilearn.image import new_img_like
from .utils import read_image, read_image_files, resize
from .nilearn_custom_utils.nilearn_utils import crop_img_to, crop_img


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)   # 得到所有图像前景并集的mask, 不是numpy数组，而是包装成了nilearn的image类型的对象。
    crop_slices = crop_img(foreground, return_slices=True, copy=True)   # crop掉前景mask为零的区域，额外在外圈保留一圈零。return_slice=True表示返回的是三维索引的切片对象列表。
    cropped = crop_img_to(foreground, crop_slices, copy=True)   # 根据上一行的crop_slices对图像foreground进行切片。
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    return crop_img(foreground, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images

# Done
def get_complete_foreground(training_data_files):
    ''' 遍历training_data_files(似乎是一个list, 每个元素又是一个list, 每个子list内存储若干图像文件的路径)，找到所有文件的前景并集，生成并返回一个mask, 前景区域为1, 背景为0。
        示例：training_data_files似乎应该类似于[[image_1_path, ..., image_m_path], [image_m+1_path, ..., image_m+k_path], ...]
    '''
    for i, set_of_files in enumerate(training_data_files):
        # 似乎set_of_files被赋值为类似[image_1_path, ..., image_m_path]这样的path list.
        subject_foreground = get_foreground_from_set_of_files(set_of_files)     # 默认情况(return_image==False)，返回ndarray类型的mask
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)     # 返回的不是numpy数组，而是做成nilearn的image类型的对象。mask维度与传进来的**单张**图像维度相同。

# Done
def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    ''' 遍历set_of_files(每个元素大概是存储文件路径的list?)，读取相应文件，找到所有文件前景区域的并集，生成并返回一个mask, 前景区域为1, 背景为0。
        示例：set_of_files似乎应该类似于[image_1_path, ..., image_m_path]
    '''
    for i, image_file in enumerate(set_of_files):
        # image_file被赋值为文件路径
        image = read_image(image_file)
        # 灰度值小于background_value - tolerance或大于background_value + tolerance的部分被认为是前景部分
        # 即认为背景灰度值在[background_value - tolerance, background_value + tolerance]范围内 
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1   # 所有模态前景部分的并集被认为是最终的前景，生成一个mask，值为1的像素对应前景部分
    if return_image:
        return new_img_like(image, foreground)  # 应该是create一个与第一个参数image同类型的新对象, 新对象存储的数据由第二个参数foreground(ndarray类型对象)给出。上同。
    else:
        return foreground

# Done
def normalize_data(data, mean, std):
    '''对shape=[N,D,H,W]的数据data执行z-score normalize.
       mean, std是N维向量，每个元素对应data中一张图片的均值和方差。
    '''
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]     # shape of mean: (N,) → (N,1,1,1), std同理
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data

# Done
def normalize_data_storage(data_storage):
    ''' 对一组数据(可以理解为一个batch)执行z-score normalize
        data_storage可能是存储若干张图片的list, 类似于[image_1, image_2, ...], 对3D图像，shape of data_storage = [N, D, H, W]
    '''
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3))) # 对axis=1,2,3的所有数据执行mean，相当于先执行data.reshape(data.size(0), -1), 然后对reshape后axis=1执行mean. 
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


