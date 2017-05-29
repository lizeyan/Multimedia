import av
from utility import *
from image_codec import *
from copy import deepcopy


def index_min_error(a, b, mask, func_error):
    error = func_error(a * mask, b * mask)
    return np.unravel_index(np.argmin(error), error.shape)


def threshold_count_error(a, b, axis=None):
    x = np.abs(a - b)
    threshold = np.median(x)
    return (x > threshold).sum(axis)


def time_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, mask: np.ndarray):
    sliding = np.concatenate(list(np.expand_dims(sliding_window(image_arr[:, :, i], BLOCK_SIZE), -1) for i in range(3)), axis=-1)
    if isinstance(mask, int):
        mask = random_selector(mask, BLOCK_SIZE[0], BLOCK_SIZE[1])
    mask = np.expand_dims(mask, -1)
    # return index_min_error(reference_block, sliding, mask, lambda a, b: np.mean(np.square(np.expand_dims(np.expand_dims(a, 0), 0) - b), axis=(-1, -2, -3)))
    return index_min_error(reference_block, sliding, mask, lambda a, b: threshold_count_error(np.expand_dims(np.expand_dims(a, 0), 0), b, (2, 3, 4)))


def frequency_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, mask: np.ndarray):
    sliding = np.concatenate(list(np.expand_dims(sliding_window(image_arr[:, :, i], BLOCK_SIZE), -1) for i in range(3)), axis=-1)
    dct_sliding = np.rollaxis(dct2(np.rollaxis(sliding, -1, 0)), 0, np.ndim(sliding))
    if isinstance(mask, int):
        mask = random_selector(mask, BLOCK_SIZE[0], BLOCK_SIZE[1])
    mask = np.expand_dims(mask, -1)
    return index_min_error(reference_block, dct_sliding, mask, lambda a, b: np.mean(np.square(np.expand_dims(np.expand_dims(a, 0), 0) - b), axis=(-1, -2, -3)))


def image2arr_pair_iterator(image_list: list, *args, **kwargs):
    last = image2arr(image_list[0], *args, **kwargs)
    for idx in range(1, len(image_list)):
        cur = image2arr(image_list[idx], *args, **kwargs)
        yield last, cur
        last = cur


def bounding_box_points(pos, size):
    result = np.ones(((size[0] + size[1]) * 2, 2), dtype=int)
    x_range = np.arange(size[0])
    y_range = np.arange(size[1])
    cur = size[0]
    result[:cur, 0] = pos[0] + x_range
    result[:cur, 1] = pos[1]

    last = cur
    cur = last + size[1]
    result[last:cur, 0] = pos[0]
    result[last:cur, 1] = pos[1] + y_range

    last = cur
    cur = last + size[0]
    result[last:cur, 0] = pos[0] + x_range
    result[last:cur, 1] = pos[1] + size[1]

    last = cur
    cur = last + size[1]
    result[last:cur, 0] = pos[0] + size[0]
    result[last:cur, 1] = pos[1] + y_range
    return result


def get_block(arr, block_pos):
    return arr[block_pos[0]: block_pos[0] + BLOCK_SIZE[0], block_pos[1]: block_pos[1] + BLOCK_SIZE[1]]


def main():
    clean_folder(PATH_OUTPUT)
    video_container = av.open(PATH_TO_VIDEO)
    max_frame_number = 40
    image_list = list([f.to_image() for f in video_container.decode(video=0)])
    log("video %s decoded" % PATH_TO_VIDEO)
    image_arr_list = list([image2arr(item) for item in image_list[:max_frame_number]])
    init_block_pos = np.array((48, 108))

    reference_block = get_block(image2arr(image_list[0]), init_block_pos)
    dct_reference_block = np.rollaxis(dct2(np.rollaxis(reference_block, -1, 0)), 0, np.ndim(reference_block))

    def work_motion_estimation(estimation_func, path_video_output):
        __output_image_arr_list = []
        __error_list = []
        for idx, __image_arr_origin in enumerate(image_arr_list):
            __image_arr = __image_arr_origin.copy()
            __block_pos = estimation_func(__image_arr)
            log("frame number: %d, block pos: (%d, %d)" % (idx + 1, __block_pos[0], __block_pos[1]))
            __bounding_box = bounding_box_points(__block_pos, BLOCK_SIZE)
            __error_list.append([np.mean(np.square(get_block(__image_arr, __block_pos), reference_block))])
            __image_arr[__bounding_box[:, 0], __bounding_box[:, 1]] = np.array([1, 0, 0])
            __output_image_arr_list.append((__image_arr * 255).astype(np.uint8))
        log("constructing video")
        output = av.open(os.path.join(PATH_OUTPUT, path_video_output), 'w')
        stream = output.add_stream("mpeg4", "10")
        stream.pix_fmt = "yuv420p"
        for img in __output_image_arr_list:
            frame = av.VideoFrame.from_ndarray(array=img, format='rgb24')
            packet = stream.encode(frame)
            output.mux(packet)
        output.close()
        return __error_list

    error_list = work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, np.ones(BLOCK_SIZE)), "time_domain.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, random_selector(2, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_random_4.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, random_selector(4, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_random_16.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, random_selector(8, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_random_64.mp4")
    print("MSE: ", np.mean(error_list))

    error_list = work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, np.ones(BLOCK_SIZE)), "frequency_domain.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 4), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_4.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 16), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_16.mp4")
    print("MSE: ", np.mean(error_list))
    error_list = work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 64), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_64.mp4")
    print("MSE: ", np.mean(error_list))


if __name__ == '__main__':
    PATH_TO_VIDEO = "cars.avi"
    PATH_OUTPUT = "lab1_output/motion_estimation"
    BLOCK_SIZE = (16, 16)
    main()
