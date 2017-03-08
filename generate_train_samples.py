import numpy as np
import cv2
import os

from PIL import Image
from matplotlib import pyplot as plt
import random

# generate rect randomly
def generate_rect(w, h):
    x1 = random.randint(0, w)
    y1 = random.randint(0, h)

    x2 = random.randint(0, w)
    y2 = random.randint(0, h)

    start_x = min(x1, x2)
    end_x = max(x1, x2)
    start_y = min(y1, y2)
    end_y = max(y1, y2)

    sample_r = [start_x, start_y, end_x, end_y]
    return sample_r

# compute intersection between gt and sample rect
# gt_rect = [top_x, top_y, down_x, down_y]
def sample_rect(gt_rect, w, h):
    # generate sample rect
    global sample_r, area_gt, area_sample
    sample_r = generate_rect(w, h)
    #compute area
    area_gt = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
    area_sample = (sample_r[2] - sample_r[0]) * (sample_r[3] - sample_r[1])
    while (area_sample < 0.3 * area_gt) or (area_sample > 1.2 * area_gt):
        sample_r = generate_rect(w, h)
        area_gt = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
        area_sample = (sample_r[2] - sample_r[0]) * (sample_r[3] - sample_r[1])

    start_x = max(sample_r[0], gt_rect[0])
    start_y = max(sample_r[1], gt_rect[1])
    end_x = min(sample_r[2], gt_rect[2])
    end_y = min(sample_r[3], gt_rect[3])
    area = (end_x - start_x) * (end_y - start_y)

    return area, sample_r

# generate samples, one image 5 negetive samples
def generate_samples(gt_rect, w, h):
    gt_area = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
    sample_list = []
    for i in range(5):
        global area, sample
        area, sample = sample_rect(gt_rect, w, h)

        # compare the intersaction area 0.3 is OK, 0,2 is too small, same images saliency map is big
        # can't generate suitable negative samples
        while area > 0.3 * gt_area:
            area, sample = sample_rect(gt_rect, w, h)
        sample_list.append(sample)

    return sample_list

def generate_pos_neg_samples(files, images_path, gt_path, save_path):
    for file in files:
        filename, suffix = os.path.splitext(file)
        print 'processing:', file
        source = Image.open(images_path + '/' + file)
        gt = Image.open(gt_path + '/' + filename + '.png')

        source_arr = np.asarray(source)
        gt_arr = np.asarray(gt)
        gt_arr = np.transpose(gt_arr)
        # print gt_arr.shape
        w, h = gt_arr.shape
        # saliency point
        sal_point = np.where(gt_arr == 255)
        max_x = max(sal_point[0])
        min_x = min(sal_point[0])
        max_y = max(sal_point[1])
        min_y = min(sal_point[1])
        # print 'min:(', min_x, min_y, ')'
        # print 'max:(', max_x, max_y, ')'

        x = [min_x, max_x, max_x, min_x]
        y = [min_y, min_y, max_y, max_y]

        gt_rect = [min_x, min_y, max_x, max_y]
        # save positive sample
        pos_region = source.crop(gt_rect)
        pos_region.save(save_path + '/pos/' + filename + '_pos.jpg')

        samples = generate_samples(gt_rect, w, h)

        # print samples
        global num
        num = 0
        # crop images
        for sample in samples:
            region_sample = source.crop(sample)
            region_sample.save(save_path + '/neg/' + filename + '_neg' + str(num) + '.jpg')
            num = num + 1


images_path = '/home/ty/data/saliency/MSRA5000/images'
save_path = '/home/ty/data/saliency/MSRA5000/neg'
gt_path = '/home/ty/data/saliency/MSRA5000/GT'

root = '/home/ty/data/saliency/MSRA5000'
files = os.listdir(images_path)
#print len(files)
#print os.path.splitext(files[0])
# total 5000, 4500 for training; 500 for validate
# training samples
generate_pos_neg_samples(files[:4500], root + '/images', root + '/GT', root + '/train')

# validation samples
generate_pos_neg_samples(files[4500:], root + '/images', root + '/GT', root + '/val')

# show images
# plt.subplot(2, 2, 1)
# plt.imshow(source)
#
# plt.plot(x, y, 'r*')
#
# plt.subplot(2, 2, 2)
# plt.imshow(gt)
# plt.plot(x, y, 'y*')
#
# plt.subplot(2, 2, 3)
# region = source.crop([min_x, min_y, max_x, max_y])
# plt.imshow(region)
#
#
# plt.subplot(2, 2, 4)
# #region_sample = source.crop(sample)
# sample_x = [samples[0][0], samples[0][2], samples[0][2], samples[0][0]]
# sample_y = [samples[0][1], samples[0][1], samples[0][3], samples[0][3]]
# plt.imshow(source)
# plt.plot(sample_x, sample_y, 'b*')
# plt.plot(x, y, 'r*')
# plt.show()

