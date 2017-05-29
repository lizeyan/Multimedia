import argparse
import re
import sys
from copy import deepcopy

from utility import *


class ImageRetriever(object):
    """
    responsible for building index and retrieving images based on the given query image
    """
    REGEX_IMAGE_LINE = re.compile(r"\s*(?P<class>[\w\d]+)/(?P<filename>[^\s]+)\s*(\d+)?\s*(\d+)?\s*")

    def __init__(self, path_to_data: str, path_to_all_images: str, bins, func_distance, center_weight):
        assert len(bins) is 3
        assert os.path.exists(path_to_data) and os.path.isdir(path_to_data), "path_to_data: %s is not valid" % path_to_data
        assert os.path.exists(path_to_all_images) and os.path.isfile(path_to_all_images), "path_to_all_images: %s is not valid" % path_to_all_images
        self.bins = tuple(bins)
        self.func_distance = func_distance
        self.center_weight = center_weight

        self.threshold = []
        self.class2code = {}
        self.class_list = []
        self.image_list = []  # list[ndarray(h * w * c)]
        self.image_name_list = []  # list[str]
        self.image_class_ndarray = []  # ndarray[int]
        self.image_vector_ndarray = None  # ndarray(n * v)

        self.setup(path_to_data, path_to_all_images)

    def setup(self, path_to_data, path_to_all_images):
        log("setting up ImageRetriever")
        for color in range(3):
            self.threshold.append(np.asarray([255 / self.bins[color] * i for i in range(1, self.bins[color])]))
        image_class_list = []
        size2image = {}
        with open(path_to_all_images) as f:
            for line in f.read().splitlines():
                match = ImageRetriever.REGEX_IMAGE_LINE.match(line)
                if not match:
                    continue
                name_class = match.group("class")
                filename = match.group("filename")
                arr = self.read_acceptable_image(os.path.join(path_to_data, name_class, filename))
                size = np.shape(arr)
                if name_class not in self.class2code:
                    self.class2code[name_class] = len(self.class_list)
                    self.class_list.append(name_class)
                if size not in size2image:
                    size2image[size] = []
                size2image[size].append(len(self.image_list))
                self.image_name_list.append(filename)
                image_class_list.append(self.class2code[name_class])
                self.image_list.append(arr)
        image_array = np.asarray(self.image_list)
        image_vector_list = []
        for group_idx in size2image.values():
            image_vector_list.append(self.image2vector(image_array[group_idx], self.threshold))
        image_vector_list = np.concatenate(image_vector_list)
        self.image_vector_ndarray = image_vector_list[np.argsort(np.concatenate(list(size2image.values())))]
        self.image_class_ndarray = np.asarray(image_class_list)
        log("finish setting up ImageRetriever")

    def query_single(self, image: Image, output_dir: str, name: str = None, size: int=30, display: bool=False) -> list:
        output_dir = os.path.join(output_dir, self.get_attributes_str())
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        distances = self.func_distance(self.image2vector(np.asarray([image]), self.threshold), self.image_vector_ndarray)
        indexes = np.argsort(distances)[:size]
        if name is not None:
            with open(os.path.join(output_dir, "res_%s.txt" % name), "w+") as f:
                for image_id in indexes:
                    print("%s/%s %f" % (self.image_class_ndarray[image_id], self.image_name_list[image_id], distances[image_id]), file=f)
                    if display:
                        log("%s/%s %f" % (self.image_class_ndarray[image_id], self.image_name_list[image_id], distances[image_id]))
        return self.image_class_ndarray[indexes]

    def query(self, image_list: list, output_dir: str, image_name_list: list = None, size: int=30, display: bool=False) -> list:
        if image_name_list is not None:
            return list(map(lambda item: self.query_single(item[0], output_dir, name=item[1], size=size, display=display), zip(image_list, image_name_list)))
        else:
            return list(map(lambda image: self.query_single(image, output_dir, size=size, display=display), image_list))

    def get_attributes_str(self):
        return "%s_%s_%s_" % self.bins + self.func_distance.__name__

    def image2vector(self, _image_list: np.ndarray, _threshold: list) -> np.ndarray:
        if np.shape(_image_list) == 3:
            image_list = np.expand_dims(_image_list, 0)
        else:
            image_list = _image_list
        assert np.ndim(image_list) == 4
        assert image_list.shape[-1] == 3
        height, width = image_list.shape[1:3]
        r_part = int(height / 6)
        c_part = int(width / 6)
        central = self.region2vector(image_list[:, r_part:height - r_part, c_part:width - c_part, :], _threshold)
        full = self.region2vector(image_list, _threshold)
        return normalize(full + central * (self.center_weight - 1))

    @staticmethod
    def region2vector(_image_list: np.ndarray, _threshold: list) -> np.ndarray:
        assert len(_threshold) is 3
        threshold = deepcopy(_threshold)
        assert np.ndim(_image_list) is 4 and np.size(_image_list, -1) is 3
        num_class = 1
        for color in range(3):
            assert np.ndim(threshold[color]) is 1
            color_min = np.min(list(map(lambda m: np.min(m[:, :, color]), _image_list)))
            color_max = np.max(list(map(lambda m: np.max(m[:, :, color]), _image_list))) + EPS
            _ = [[color_min], np.asarray(threshold[color]), [color_max]]
            threshold[color] = np.concatenate(_)
            threshold[color] = np.sort(threshold[color])
            num_class *= len(threshold[color]) - 1
        image_list = np.zeros(_image_list.shape)
        for color in range(3):
            image_list[:, :, :, color] = np.searchsorted(threshold[color], _image_list[:, :, :, color]) - 1
        image_list = image_list[:, :, :, 0] + image_list[:, :, :, 1] * (len(threshold[0]) - 1) + image_list[:, :, :, 2] * (len(threshold[0]) - 1) * (len(threshold[1]) - 1)
        return np.asarray([[np.count_nonzero(m == c) for c in range(num_class)] for m in image_list], dtype=np.float64)

    @staticmethod
    def read_acceptable_image(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))


def get_query_set(path_to_query: str, base_dir: str) -> tuple:
    image_list = []
    name_class_list = []
    filename_list = []
    with open(path_to_query) as f:
        for line in f.readlines():
            match = ImageRetriever.REGEX_IMAGE_LINE.match(line)
            if not match:
                continue
            name_class = match.group("class")
            filename = match.group("filename")
            image_list.append(ImageRetriever.read_acceptable_image(os.path.join(base_dir, name_class, filename)))
            name_class_list.append(name_class)
            filename_list.append(filename)
    return image_list, name_class_list, filename_list


def main(flags):
    retriever = ImageRetriever(flags.base_dir, flags.data, bins=flags.bins, func_distance=flags.func_distance, center_weight=flags.regional)
    log("attributes:", retriever.get_attributes_str())
    if flags.query is not None:
        image_list, ground_truth_list, filename_list = get_query_set(flags.query, flags.base_dir)
        ground_truth_code_list = np.asarray(list(retriever.class2code[item] for item in ground_truth_list))
        predict_class_code_list = np.asarray(retriever.query(image_list, flags.output_dir, image_name_list=filename_list, size=flags.length))
        total_precision = np.count_nonzero(np.expand_dims(ground_truth_code_list, 1) == predict_class_code_list) / np.size(predict_class_code_list)
        log("total precision:", total_precision)
    else:
        for line in sys.stdin:
            match = retriever.REGEX_IMAGE_LINE.match(line)
            if match:
                name_class = match.group("class")
                filename = match.group("filename")
                image = ImageRetriever.read_acceptable_image(os.path.join(flags.base_dir, name_class, filename))
                predict_class_code_list = np.asarray(retriever.query_single(image, flags.output_dir, filename, size=flags.length, display=True))
                ground_truth_code_list = np.asarray([retriever.class2code[name_class]])
                precision = np.count_nonzero(ground_truth_code_list == predict_class_code_list) / np.size(predict_class_code_list)
                log("precision:", precision)
    return


def argparse_dir(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid dir path" % path)


def argparse_file(path: str):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid file path" % path)


def argparse_bins(s: str) -> int:
    try:
        return int(s)
    except Exception:
        raise argparse.ArgumentTypeError("invalid bins setting")


def argparse_func_distance(s: str):
    if s in name2func_distance:
        return name2func_distance[s]
    else:
        raise argparse.ArgumentTypeError("invalid distance function")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageRetriever for the specified parameter. The full work flow is run_lab2.py")
    parser.add_argument("--base_dir", "-b", help="base path of all images", default="./DataSet", type=argparse_dir)
    parser.add_argument("--output_dir", "-o", help="output dir path of all images", default="./lab2_output", type=argparse_dir)
    parser.add_argument("--data", "-d", help="text file contains queried images", default="./AllImages.txt", type=argparse_file)
    parser.add_argument("--query", "-q", help="text file contains query images", type=argparse_file)
    parser.add_argument("--func_distance", "-f", help="which distance is to be used", type=argparse_func_distance)
    parser.add_argument("--bins", "-s", help="bins setting, using space separated integers", nargs=3, type=argparse_bins)
    parser.add_argument("--length", "-l", help="the number of chosen images", default=30, type=int)
    parser.add_argument("--regional", "-r", help="regional based Image Retrieval. Central block is more important", default=1, type=float)
    main(parser.parse_args())
