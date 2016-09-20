from src.tools.simulator_data_parser import DatParser
import numpy as np
import os
import tarfile
import random
import string


class DatasetHelper:
    #
    # DIMENSIONI:
    # x_set:      ( #campioni, #feature )
    # x_labels:   ( #campioni, #classi )
    #

    LABEL_STROKE = 1
    LABEL_HEALTHY = 0

    def __init__(self, path):
        self.path = path
        self.dummy = 0

    def foo(self):
        print("bar")

    @staticmethod
    def load_archive(filename, symmetry):
        """
        Loads a dataset archive. Reads the filename provided as parameter and returns the dataset as ndarray.
        The returned dataset dimensions are (#samples, #features)
        :param filename: the dataset filename
        :param symmetry: builds symmetry features
        :return:
        """
        char_set = string.ascii_uppercase + string.digits
        with tarfile.open(filename) as tar:
            path = "tmp/{}/".format(''.join(random.sample(char_set * 6, 6)))
            tar.extractall(path)
            dataset = DatasetHelper.load_data(path, symmetry)
            os.system("rm {} -rf".format(path))
            return dataset

    @staticmethod
    def load_data(path, symmetry):
        """
        Loads a dataset. Reads into the path provided as parameter and returns the dataset as ndarray.
        The returned dataset dimensions are (#samples, #features)
        :param path: folder to look into
        :param symmetry: builds symmetry features
        :return: the dataset
        """

        files = [f for f in os.listdir(path)]
        # number_of_files = len(files)

        number_of_views = DatParser.count_views("{}/{}".format(path, files[0]))

        first = True

        for f in files:
            filepath = "{}/{}".format(path, f)
            views_no = DatParser.count_views(filepath)
            if views_no != number_of_views:
                raise ValueError("Mismatch in number of views in {}! check .dat files!".format(path))
            if first:
                if symmetry:
                    dataset = DatParser.parse_file_and_compute_symmetry(filepath)
                else:
                    dataset = DatParser.parse_file(filepath)
                first = False
            else:
                if symmetry:
                    dataset = np.row_stack((dataset, DatParser.parse_file_and_compute_symmetry(filepath)))
                else:
                    dataset = np.row_stack((dataset, DatParser.parse_file(filepath)))

        return dataset

    @staticmethod
    def generate_labels(samples, label):
        """
        Generates a label set.
        Each label is a one-hot array, i.e. , for the n-th sample, if LABEL_HEALTHY
         label =    [ 1  0 ]
        else, if LABEL_STROKE
         label =    [ 0  1 ]

        :param samples: Number of samples in dataset.
        :param label: Ground truth, can be either "LABEL_STROKE" or "LABEL_HEALTHY" (static constants of this class)
        :return: The labels tensor.
        """
        if label == DatasetHelper.LABEL_HEALTHY:
            return np.column_stack((np.zeros((samples, 1)), np.ones((samples, 1))))
        else:
            return np.column_stack((np.ones((samples, 1)), np.zeros((samples, 1))))
