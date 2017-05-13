import os
from copy import deepcopy
import re
from utility import *
import argparse


class ImageRetriever(object):
    """
    responsible for building index and retrieving images based on the given query image
    """
    REGEX_IMAGE_LINE = re.compile(r"\s*(?P<class>[\w\d]+)/(?P<filename>[^\s]+)\s*(?P<height>\d+)\s*(?P<width>\d+)\s*")

    def __init__(self, path_to_data: str, path_to_all_images: str, bins: tuple=(2, 4, 2), func_distance=euclidean_distance):
        assert len(bins) is 3
        assert os.path.exists(path_to_data) and os.path.isdir(path_to_data), "path_to_data: %s is not valid" % path_to_data
        assert os.path.exists(path_to_all_images) and os.path.isfile(path_to_all_images), "path_to_all_images: %s is not valid" % path_to_all_images
        self.bins = bins
        self.func_distance = func_distance

        self.threshold = []
        self.class2code = {}
        self.class_list = []
        self.image_list = []  # list[ndarray(h * w * c)]
        self.image_name_list = []  # list[str]
        self.image_class_list = []  # ndarray[int]
        self.image_vector_list = None  # ndarray(n * v)

        self.setup(path_to_data, path_to_all_images)

    def setup(self, path_to_data, path_to_all_images):
        log("setting up ImageRetriever")
        for color in range(3):
            self.threshold.append(np.asarray([255 / self.bins[color] * i for i in range(1, self.bins[color])]))

        size2image = {}

        with open(path_to_all_images) as f:
            for line in f.read().splitlines():
                match = ImageRetriever.REGEX_IMAGE_LINE.match(line)
                if not match:
                    continue
                name_class = match.group("class")
                filename = match.group("filename")
                height = int(match.group("height"))
                width = int(match.group("width"))
                size = (height, width)
                if name_class not in self.class2code:
                    self.class2code[name_class] = len(self.class_list)
                    self.class_list.append(name_class)
                if size not in size2image:
                    size2image[size] = []
                size2image[size].append(len(self.image_list))
                self.image_name_list.append(filename)
                self.image_class_list.append(self.class2code[name_class])
                self.image_list.append(self.read_acceptable_image(os.path.join(path_to_data, name_class, filename)))
        image_array = np.asarray(self.image_list)
        image_vector_list = []
        for group_idx in size2image.values():
            image_vector_list.append(self.image2vector(image_array[group_idx], self.threshold))
        image_vector_list = np.concatenate(image_vector_list)
        self.image_vector_list = image_vector_list[np.argsort(np.concatenate(list(size2image.values())))]
        self.image_class_list = np.asarray(self.image_class_list)
        log("finish setting up ImageRetriever")

    def query_single(self, image: Image, size: int=30) -> list:
        distances = self.func_distance(self.image2vector(np.asarray([image]), self.threshold), self.image_vector_list)
        return self.image_class_list[np.argsort(distances)[:size]]

    def query(self, image_list: list, size: int=30) -> list:
        return list(map(lambda image: self.query_single(image, size), image_list))

    @staticmethod
    def image2vector(image_list: np.ndarray, _threshold: list) -> np.ndarray:
        assert len(_threshold) is 3
        threshold = deepcopy(_threshold)
        assert np.ndim(image_list) is 4 and np.size(image_list, -1) is 3
        num_class = 1
        for color in range(3):
            assert np.ndim(threshold[color]) is 1
            color_min = np.min(list(map(lambda m: np.min(m[:, :, color]), image_list)))
            color_max = np.max(list(map(lambda m: np.max(m[:, :, color]), image_list))) + EPS
            _ = [[color_min], np.asarray(threshold[color]), [color_max]]
            threshold[color] = np.concatenate(_)
            threshold[color] = np.sort(threshold[color])
            num_class *= len(threshold[color]) - 1
        for color in range(3):
            image_list[:, :, :, color] = np.searchsorted(threshold[color], image_list[:, :, :, color])
        image_list = image_list[:, :, :, 0] + image_list[:, :, :, 1] * 3 + image_list[:, :, :, 2] * 9
        return np.asarray([[np.count_nonzero(m == c) for c in range(num_class)] for m in image_list])

    @staticmethod
    def read_acceptable_image(path: str) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))


def get_query_set(path_to_query: str) -> tuple:
    image_list = []
    name_class_list = []
    filename_list = []
    with open(path_to_query) as f:
        for line in f.read().splitlines():
            match = ImageRetriever.REGEX_IMAGE_LINE.match(line)
            if not match:
                continue
            name_class = match.group("class")
            filename = match.group("filename")
            image_list.append(ImageRetriever.read_acceptable_image(os.path.join(FLAGS.base_dir, name_class, filename)))
            name_class_list.append(name_class)
            filename_list.append(filename)
    return image_list, name_class_list, filename_list


def main():
    retriever = ImageRetriever(FLAGS.base_dir, FLAGS.data, bins=(2, 4, 2), func_distance=euclidean_distance)
    image_list, ground_truth_list, filename_list = get_query_set(FLAGS.query)
    ground_truth_code_list = np.asarray(list(retriever.class2code[item] for item in ground_truth_list))
    predict_class_code_list = np.asarray(retriever.query(image_list, 5))
    total_precision = np.count_nonzero(np.expand_dims(ground_truth_code_list, 1) == predict_class_code_list) / np.size(predict_class_code_list)
    print("total precision:", total_precision)
    return


def valid_dir(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid dir path" % path)


def valid_file(path: str):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("%s is not a valid file path" % path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageRetriever")
    parser.add_argument("--base_dir", "-b", help="base path of all images", default="./DataSet", type=valid_dir)
    parser.add_argument("--data", "-d", help="text file contains queried images", default="./AllImages.txt", type=valid_file)
    parser.add_argument("--query", "-q", help="text file contains query images", default="./QueryImages.txt", type=valid_file)
    FLAGS = parser.parse_args()
    main()
