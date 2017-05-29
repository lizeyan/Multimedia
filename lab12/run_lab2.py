import argparse
import re
import shlex
import subprocess

from utility import *

bins_options = [(2, 2, 2), (4, 2, 2), (2, 4, 2), (2, 2, 4), (4, 8, 4), (8, 8, 8)]
# bins_options = [(2, 4, 2)]
func_distance_options = list(name2func_distance.keys())
# func_distance_options = ["l1", "bh", "hi", "l2"]
center_weight_options = [0.5, 1, 4, 16, 256]


class FLAGS:
    base_dir = "./DataSet"
    output_dir = "./lab2_output"
    data = "./AllImages.txt"
    query = "./QueryImages.txt"
    func_distance = None
    bins = None
    length = 30


def iter_on_bins_distance(center_weight):
    children = []
    index = []
    result_matrix = np.zeros(shape=(len(func_distance_options), len(bins_options)))
    active_count = 0
    is_running = []

    for x, func_distance in enumerate(func_distance_options):
        for y, bins in enumerate(bins_options):
            cmd = "python image_retrieval.py -q QueryImages.txt -s %d %d %d " % bins + " -f %s " % func_distance + " -r %f" % center_weight
            while active_count == FLAGS.max_worker:
                time.sleep(1)
                for idx, child in enumerate(children):
                    if is_running[idx] and child.poll() is not None:
                        active_count -= 1
                        is_running[idx] = False
            children.append(subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE))
            log("start running command: \"%s\"" % cmd)
            active_count += 1
            is_running.append(True)
            index.append((x, y))
    # print each progress's output
    regex_precision = re.compile(r"total\s*precision\s*:\s*(?P<precision>[\d\.]+)")
    for idx, child in zip(index, children):
        child.wait()
        print("===================================")
        output = child.stdout.read().decode('utf-8')
        for line in output.splitlines():
            print(line)
            match = regex_precision.search(line)
            if match:
                result_matrix[idx] = eval(match.group("precision"))

    # print result table
    result = "distance" + "".join([",\"%d:%d:%d\"" % item for item in bins_options]) + "\n"
    for func_name, l in zip(func_distance_options, result_matrix.tolist()):
        result += "%s" % func_name + "".join([",%f" % p for p in l]) + "\n"
    with open("lab2_result_%s.csv" % str(center_weight), "w+") as f:
        print(result, file=f)
    print(result)


def main():
    for center_weight in center_weight_options:
        iter_on_bins_distance(center_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="full work flow for lab2")
    parser.add_argument("--max_worker", "-j", help="maximum number of process to be used", default=4, type=int)
    FLAGS = parser.parse_args()
    main()
