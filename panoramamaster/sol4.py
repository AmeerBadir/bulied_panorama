# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite
from scipy import signal
import sol4_utils

CURLEVEL = 0
THIRD_LEV = 2


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    y_derivative_img = signal.convolve2d(im, np.array([[1], [0], [-1]]), mode='same', boundary='symm')
    x_derivative_img = signal.convolve2d(im, np.array([[1, 0, -1]]), mode='same', boundary='symm')
    i_x2 = x_derivative_img * x_derivative_img
    i_x2_blur = sol4_utils.blur_spatial(i_x2, 3)
    i_y2 = y_derivative_img ** 2
    i_y2_blur = sol4_utils.blur_spatial(i_y2, 3)
    i_xy = x_derivative_img * y_derivative_img
    i_xy_blur = sol4_utils.blur_spatial(i_xy, 3)
    r = (i_x2_blur * i_y2_blur - (i_xy_blur ** 2)) - (((i_x2_blur + i_y2_blur) ** 2) * 0.04)
    corn = np.where(non_maximum_suppression(r))
    return np.column_stack((corn[1], corn[0]))


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """

    descriptor = np.empty((pos.shape[0], 1 + (desc_rad * 2), 1 + (desc_rad * 2)), dtype=np.float64)
    j = 0
    lst1 = []
    for cord in pos:
        patch1_start = cord[0] - desc_rad
        patch1_end = cord[0] + desc_rad + 1
        patch2_start = cord[1] - desc_rad
        patch2_end = cord[1] + desc_rad + 1
        rec = np.meshgrid(np.arange(patch2_start, patch2_end), np.arange(patch1_start, patch1_end))
        norm_pixels = map_coordinates(im, rec, order=1, prefilter=False).T
        euclidean_norm = np.linalg.norm((norm_pixels - np.mean(norm_pixels)))
        lst1.append(((norm_pixels - np.mean(norm_pixels)) / euclidean_norm) if euclidean_norm != 0 else euclidean_norm)
        descriptor[j, :, :] = (
                    (norm_pixels - np.mean(norm_pixels)) / euclidean_norm) if euclidean_norm != 0 else euclidean_norm
        j += 1
    return descriptor


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    feature_point = spread_out_corners(pyr[0], 7, 7, 3)
    return feature_point, sample_descriptor(pyr[2], (feature_point * 0.25).astype(np.float64), 3)


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    sh1 = desc1.shape
    sh2 = desc2.shape[0]
    new_len = sh1[2] * sh1[2]
    s = np.dot(np.reshape(desc1, (sh1[0], new_len)), (np.reshape(desc2, (sh2, new_len))).T)
    len1 = s.shape
    l = 0
    idx_d1 = []
    idx_d2 = []
    while l != len1[0]:
        best2 = np.where(s[l] - np.max(s[l]) == 0)
        help_lst = [idx_d1, idx_d2]
        idx_d1, idx_d2 = match_helper(min_score, help_lst, best2[0], s, l)
        l += 1
    arr1 = np.array(idx_d1)
    arr2 = np.array(idx_d2)
    return arr1.astype(np.int), arr2.astype(np.int)


def match_helper(min_score, help_lst, best2, s, i):

    for j in best2:
        max_ = np.amax(s[:, j])
        if s[i][j] > min_score:
            if s[i][j] != max_:
                continue
            else:
                help_lst[1].append(j)
                help_lst[0].append(i)
    return help_lst[0], help_lst[1]


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """

    pos1 = np.insert(pos1, pos1.shape[1], 1, axis=1)
    pos2 = np.zeros((pos1.shape[0], pos1.shape[1] + 1))
    pos1_lst = [pos1[:, 0], pos1[:, 1], pos1[:, 2]]
    lst1 = []
    for i in range(3):
        sum_h = H12[i, 0] * pos1_lst[0] + H12[i, 1] * pos1_lst[1] + H12[i, 2] * pos1_lst[2]
        lst1.append(sum_h)
    for i in range(3):
        pos2[:, i] = lst1[i]
    div_factor = pos2[:, 2]
    for i in range(2):
        pos2[:, i] /= div_factor
    final_hom = pos2[:, [0, 1]]
    return final_hom


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    inlier_matches = []
    num_matches = 0
    if num_iter != 0:
        pick_rand = np.random.choice(points1.shape[0], 2)
        num_matches, inlier_matches = calculate_euclidean_and_inlier_match(points1, points2, inlier_tol, pick_rand[0],
                                                                           pick_rand[1], translation_only)
    j = 1
    while j != num_iter:
        pick_rand = np.random.choice(points1.shape[0], 2)
        get_inlier_and_matches = calculate_euclidean_and_inlier_match(points1, points2, inlier_tol, pick_rand[0],
                                                                      pick_rand[1], translation_only)
        if get_inlier_and_matches[0] > num_matches:
            inlier_matches = get_inlier_and_matches[1]
            num_matches = get_inlier_and_matches[1].shape[0]
        j += 1
    if num_iter == 0:
        matches = np.where([])
        return [estimate_rigid_transform(points1[matches], points2[matches], translation_only), matches[0]]
    return [estimate_rigid_transform(points1[inlier_matches], points2[inlier_matches], translation_only),
            inlier_matches]


def calculate_euclidean_and_inlier_match(points1, points2, inlier_tol, rand1, rand2, translation_only):
    homography = estimate_rigid_transform(np.array([points1[rand1], points1[rand2]]),
                                          np.array([points2[rand1], points2[rand2]]), translation_only)
    diff = apply_homography(points1, homography) - points2
    euclide_diff = np.linalg.norm(diff, axis=1) * np.linalg.norm(diff, axis=1)
    euclide_diff = euclide_diff.reshape(points1.shape[0], 1)
    inlier = np.where(euclide_diff < inlier_tol)
    inlier = inlier[0]
    return inlier.shape[0], inlier


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    p1_shape = points1.shape[0]
    plt.figure()
    # display red matches
    p1_x_y = points1[:, 0], points1[:, 1]
    plt.plot(p1_x_y[0], p1_x_y[1], 'ro', markersize=1, color='r', linewidth=0.5)
    strech_p2_x = points2[:, 0] + p1_shape
    p2_x_y = strech_p2_x, points2[:, 1]
    plt.plot(p2_x_y[0], p2_x_y[1], 'ro', markersize=1, color='r', linewidth=0.5)
    # display outliers
    p2 = points2.copy()
    p2[:, 0] += im1.shape[1]
    remove_inlier = np.append(np.delete(points1, inliers, axis=0), np.delete(p2, inliers, axis=0), axis=1)
    x1 = remove_inlier[:, 0]
    x2 = remove_inlier[:, 2]
    y1 = remove_inlier[:, 1]
    y2 = remove_inlier[:, 3]
    lst1 = [x1, x2]
    lst2 = [y1, y2]
    plt.plot(lst1, lst2, markerfacecolor='r', c='b', linewidth=0.2, markersize=3, marker='s')

    # display inlines
    i2_inline = points2[inliers]
    x_yellow = [points1[inliers][:, 0], i2_inline[:, 0] + p1_shape]
    y_yellow = [points1[inliers][:, 1], i2_inline[:, 1]]
    plt.plot(x_yellow, y_yellow, color='yellow', linewidth=0.2)
    # display connected images
    plt.imshow(np.hstack((im1, im2)), cmap='gray')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    len_homography = len(H_succesive) + 1
    ret_homography = []
    identity_mat = np.eye(3)
    for i in range(len(H_succesive)):
        if i == len(H_succesive) - 1:
            ret_homography.append(identity_mat)
            ret_homography.append(identity_mat)
        elif i != len(H_succesive) - 1:
            ret_homography.append(identity_mat)
    # remeber -HI,M = -HI+1,M * HI,I + 1
    i = 0
    while i != len_homography:
        if i > m:
            h_tag = ret_homography[i - 1]
            h = np.dot(h_tag, np.linalg.inv(H_succesive[i - 1]))
            norm = h / h[2, 2]
            ret_homography[i] = norm
        elif i - m <= 0:
            if i == 0:
                apply_hom = np.dot(ret_homography[m], H_succesive[m - 1])
                ret_homography[m - 1] = apply_hom / apply_hom[2, 2]
            else:
                hi_tag = ret_homography[m - i]
                homography_j = H_succesive[m - i - 1]
                apply_one_homography = np.dot(hi_tag, homography_j)
                norm = apply_one_homography / apply_one_homography[2, 2]
                ret_homography[m - i - 1] = norm
        i += 1
    return ret_homography


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    left_side = [0, 0], [0, h - 1]
    right_side = [w - 1, 0], [w - 1, h - 1]
    img_corners = np.array([left_side[0], right_side[0], left_side[1], right_side[1]])
    h_coords = apply_homography(img_corners, homography)
    x_cord = h_coords[:, 0]
    y_cord = h_coords[:, 1]
    ret_arr = np.array([[np.min(x_cord), np.min(y_cord)], [np.max(x_cord), np.max(y_cord)]])
    return ret_arr.astype(np.int)


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    shape_img = image.shape
    h1 = homography / homography[2, 2]
    bounding_box1, bound2 = compute_bounding_box(h1, shape_img[1], shape_img[0])
    canvass = np.meshgrid(np.arange(bounding_box1[0], bound2[0] + 1), np.arange(bounding_box1[1], bound2[1] + 1))
    canvas_x_shape = canvass[0].shape
    f1 = canvass[0].flatten()
    f1 = f1.reshape(-1, 1)
    f2 = canvass[1].flatten()
    f2 = f2.reshape(-1, 1)
    to_wrap_back = np.hstack((f1, f2))
    invers = np.linalg.inv(homography / homography[2, 2])
    wrapped = apply_homography(to_wrap_back, invers)
    w = wrapped[:, :2]
    wrapped_positions = [w[1], w[0]]
    result = map_coordinates(image, wrapped_positions, order=1)
    result = result.reshape(canvas_x_shape)
    return result


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):

        """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
