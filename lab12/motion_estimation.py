from matplotlib import pyplot as plt
from matplotlib import patches as patches
import av
from utility import *
from image_codec import *


def index_min_error(a, b, func_error):
    error = func_error(a, b)
    return np.unravel_index(np.argmin(error), error.shape)


def threshold_count_error(a, b, threshold=1e-2, axis=None):
    x = np.abs(a - b)
    return (x > threshold).sum(axis) / np.size(x)


def center_weight_mse_error(a, b, mask, weight, axis=None):
    error = np.square(np.expand_dims(np.expand_dims(a, 0), 0) - b) * np.expand_dims(mask, -1)
    center = error * np.expand_dims(center_selector(1/4, np.size(a, -3), np.size(a, -2)), -1)
    return np.mean(error + center * (weight - 1.0), axis=axis)


def center_weight_time_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, mask: np.ndarray, weight=1.0):
    sliding = np.rollaxis(sliding_window(np.rollaxis(image_arr, -1, 0), BLOCK_SIZE), 0, np.ndim(image_arr) + len(BLOCK_SIZE))
    return index_min_error(reference_block, sliding, lambda a, b: center_weight_mse_error(a, b, mask, weight, axis=(-1, -2, -3)))


def counter_time_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, threshold=1e-2):
    sliding = np.rollaxis(sliding_window(np.rollaxis(image_arr, -1, 0), BLOCK_SIZE), 0, np.ndim(image_arr) + len(BLOCK_SIZE))
    return index_min_error(reference_block, sliding, lambda a, b: threshold_count_error(np.expand_dims(np.expand_dims(a, 0), 0), b, threshold, (-3, -2, -1)))


def time_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, mask: np.ndarray):
    sliding = np.rollaxis(sliding_window(np.rollaxis(image_arr, -1, 0), BLOCK_SIZE), 0, np.ndim(image_arr) + len(BLOCK_SIZE))
    return index_min_error(reference_block, sliding, lambda a, b: np.mean(np.square(np.expand_dims(np.expand_dims(a, 0), 0) - b) * np.expand_dims(mask, -1), axis=(-1, -2, -3)))


def frequency_domain_block_matching(reference_block: np.ndarray, image_arr: np.ndarray, mask: np.ndarray):
    sliding = np.rollaxis(sliding_window(np.rollaxis(image_arr, -1, 0), BLOCK_SIZE), 0, np.ndim(image_arr) + len(BLOCK_SIZE))
    dct_sliding = np.rollaxis(dct2(np.rollaxis(sliding, -1, 0)), 0, np.ndim(sliding))
    return index_min_error(reference_block, dct_sliding, lambda a, b: np.mean(np.square(np.expand_dims(np.expand_dims(a, 0), 0) - b) * np.expand_dims(mask, -1), axis=(-1, -2, -3)))


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
    video_container = av.open(PATH_TO_VIDEO)
    max_frame_number = 45
    image_list = list([f.to_image() for f in video_container.decode(video=0)])
    log("video %s decoded" % PATH_TO_VIDEO)
    image_arr_list = list([image2arr(item) for item in image_list[:max_frame_number]])
    init_block_pos = np.array((48, 108))

    reference_block = get_block(image2arr(image_list[0]), init_block_pos)
    dct_reference_block = np.rollaxis(dct2(np.rollaxis(reference_block, -1, 0)), 0, np.ndim(reference_block))

    def work_motion_estimation(estimation_func, indicator):
        frames_output_path = os.path.join(PATH_OUTPUT, indicator)
        clean_folder(frames_output_path)
        __error_list = []
        output = av.open(os.path.join(PATH_OUTPUT, indicator + ".mp4"), 'w')
        stream = output.add_stream("mpeg4", "10")
        stream.pix_fmt = "yuv420p"
        __last_pos = init_block_pos
        for idx, __image_arr in enumerate(image_arr_list):
            __block_pos = estimation_func(__image_arr)
            log("[%s]" % indicator, "frame number: %d, block pos: (%d, %d)" % (idx + 1, __block_pos[0], __block_pos[1]))
            __error_list.append([np.mean(np.square(get_block(__image_arr, __block_pos) - reference_block))])

            # plot
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(__image_arr, cmap=plt.get_cmap("bone"))
            ax.add_patch(patches.Rectangle((__block_pos[1], __block_pos[0]), BLOCK_SIZE[0], BLOCK_SIZE[1], fill=None, color="g"))
            ax.arrow(__last_pos[1], __last_pos[0], __block_pos[1] - __last_pos[1], __block_pos[0] - __last_pos[0], fc='k', ec='k')
            plt.annotate('', xy=(init_block_pos[1], init_block_pos[0]), xycoords='data', xytext=(__block_pos[1], __block_pos[0]), textcoords='data', arrowprops={'arrowstyle': '<-', "color": "r"})
            __last_pos = __block_pos
            fig.canvas.draw()
            image_arr_new = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image_arr_new = image_arr_new.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            Image.fromarray(image_arr_new, "RGB").save(os.path.join(frames_output_path, "%d.jpg" % (idx + 1)))
            plt.close(fig)
            # to video
            frame = av.VideoFrame.from_ndarray(array=image_arr_new, format='rgb24')
            packet = stream.encode(frame)
            output.mux(packet)
        output.close()
        plt.clf()
        plt.plot(np.arange(1, len(image_arr_list) + 1), __error_list)
        plt.title("%s-MSE" % indicator)
        plt.xlabel("frame number")
        plt.ylabel("MSE")
        plt.savefig(os.path.join(PATH_OUTPUT, "%s_MSE-frame_number.png" % indicator))
        log("[%s]" % indicator, "Average MSE:", np.mean(__error_list))

    work_motion_estimation(lambda x: center_weight_time_domain_block_matching(reference_block, x, center_selector(1/2, BLOCK_SIZE[0], BLOCK_SIZE[1]), 8), "center_weight_8")
    work_motion_estimation(lambda x: center_weight_time_domain_block_matching(reference_block, x, center_selector(1/2, BLOCK_SIZE[0], BLOCK_SIZE[1]), 2), "center_weight_2")
    work_motion_estimation(lambda x: center_weight_time_domain_block_matching(reference_block, x, center_selector(1/2, BLOCK_SIZE[0], BLOCK_SIZE[1]), 4), "center_weight_4")
    work_motion_estimation(lambda x: counter_time_domain_block_matching(reference_block, x, 1e-1), "counter_1e-1")
    work_motion_estimation(lambda x: counter_time_domain_block_matching(reference_block, x, 1e-2), "counter_1e-2")
    work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, np.ones(BLOCK_SIZE)), "time_domain")
    work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, center_selector(1/2, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_1_4")
    work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, center_selector(1/4, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_1_16")
    work_motion_estimation(lambda x: time_domain_block_matching(reference_block, x, center_selector(1/8, BLOCK_SIZE[0], BLOCK_SIZE[1])), "time_domain_1_64")
    work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, np.ones(BLOCK_SIZE)), "frequency_domain")
    work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 4), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_4")
    work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 16), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_16")
    work_motion_estimation(lambda x: frequency_domain_block_matching(dct_reference_block, x, zig_zag_selector(int(BLOCK_SIZE[0] * BLOCK_SIZE[1] / 64), BLOCK_SIZE[0], BLOCK_SIZE[1])), "frequency_domain_zig_zag_64")


if __name__ == '__main__':
    PATH_TO_VIDEO = "cars.avi"
    PATH_OUTPUT = "lab1_output/motion_estimation"
    BLOCK_SIZE = (16, 16)
    main()
