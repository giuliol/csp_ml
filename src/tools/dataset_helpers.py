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
        import os
        with tarfile.open(filename) as tar:
            path = "tmp/{}/".format(''.join(random.sample(char_set * 6, 6)))
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path)
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
                    dataset = DatParser.parse_file(filepath, 0)
                first = False
            else:
                if symmetry:
                    dataset = np.row_stack((dataset, DatParser.parse_file_and_compute_symmetry(filepath)))
                else:
                    dataset = np.row_stack((dataset, DatParser.parse_file(filepath, 0)))

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
        if label == DatasetHelper.LABEL_STROKE:
            return np.column_stack((np.zeros((samples, 1)), np.ones((samples, 1))))
        else:
            return np.column_stack((np.ones((samples, 1)), np.zeros((samples, 1))))


class QuickLoader:
    def __init__(self):
        dummy = 0

    @staticmethod
    def load_archives_training(healthy_training_filepath, stroke_training_filepath, symmetry):
        print("healthy training set {}\nstroke training set {}".format(
            healthy_training_filepath,
            stroke_training_filepath
        ))

        healthy_training_set = DatasetHelper.load_archive(healthy_training_filepath, symmetry)
        healthy_training_labels = DatasetHelper.generate_labels(healthy_training_set.shape[0],
                                                                DatasetHelper.LABEL_HEALTHY)

        stroke_training_set = DatasetHelper.load_archive(stroke_training_filepath, symmetry)
        stroke_training_labels = DatasetHelper.generate_labels(stroke_training_set.shape[0],
                                                               DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        return training_set, training_labels

    @staticmethod
    def load_archives(healthy_training_filepath, healthy_test_filepath, stroke_training_filepath, stroke_test_filepath):
        print("healthy training set {}\nhealthy test set {}\nstroke training set {}\nstroke test set {}".format(
            healthy_training_filepath,
            healthy_test_filepath,
            stroke_training_filepath,
            stroke_test_filepath
        ))

        healthy_training_set = DatasetHelper.load_archive(healthy_training_filepath, 1)
        healthy_training_labels = DatasetHelper.generate_labels(healthy_training_set.shape[0],
                                                                DatasetHelper.LABEL_HEALTHY)
        healthy_test_set = DatasetHelper.load_archive(healthy_test_filepath, 1)
        healthy_test_labels = DatasetHelper.generate_labels(healthy_test_set.shape[0], DatasetHelper.LABEL_HEALTHY)

        stroke_training_set = DatasetHelper.load_archive(
            stroke_training_filepath, 1)
        stroke_training_labels = DatasetHelper.generate_labels(stroke_training_set.shape[0], DatasetHelper.LABEL_STROKE)
        stroke_test_set = DatasetHelper.load_archive(stroke_test_filepath, 1)
        stroke_test_labels = DatasetHelper.generate_labels(stroke_test_set.shape[0], DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))
        test_set = np.row_stack((healthy_test_set, stroke_test_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set4_and_5():
        healthy_training_set4, healthy_training_labels4, healthy_test_set4, healthy_test_labels4 = DatasetHelper.load_archive(
            "res/set_4/healthy_training.tar.gz", 1), \
                                                                                                   DatasetHelper.generate_labels(
                                                                                                       1600,
                                                                                                       DatasetHelper.LABEL_HEALTHY), \
                                                                                                   DatasetHelper.load_archive(
                                                                                                       "res/set_4/healthy_test.tar.gz",
                                                                                                       1), \
                                                                                                   DatasetHelper.generate_labels(
                                                                                                       400,
                                                                                                       DatasetHelper.LABEL_HEALTHY)

        stroke_training_set4, stroke_training_labels4, stroke_test_set4, stroke_test_labels4 = DatasetHelper.load_archive(
            "res/set_4/stroke_training.tar.gz", 1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   1600,
                                                                                                   DatasetHelper.LABEL_STROKE), \
                                                                                               DatasetHelper.load_archive(
                                                                                                   "res/set_4/stroke_test.tar.gz",
                                                                                                   1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   400,
                                                                                                   DatasetHelper.LABEL_STROKE)

        training_set4 = np.row_stack((healthy_training_set4, stroke_training_set4))
        test_set4 = np.row_stack((healthy_test_set4, stroke_test_set4))

        training_labels4 = np.row_stack((healthy_training_labels4, stroke_training_labels4))
        test_labels4 = np.row_stack((healthy_test_labels4, stroke_test_labels4))

        healthy_training_set5, healthy_training_labels5, healthy_test_set5, healthy_test_labels5 = DatasetHelper.load_archive(
            "res/set_5/healthy_training.tar.gz", 1), \
                                                                                                   DatasetHelper.generate_labels(
                                                                                                       700,
                                                                                                       DatasetHelper.LABEL_HEALTHY), \
                                                                                                   DatasetHelper.load_archive(
                                                                                                       "res/set_4/healthy_test.tar.gz",
                                                                                                       1), \
                                                                                                   DatasetHelper.generate_labels(
                                                                                                       300,
                                                                                                       DatasetHelper.LABEL_HEALTHY)

        stroke_training_set5, stroke_training_labels5, stroke_test_set5, stroke_test_labels5 = DatasetHelper.load_archive(
            "res/set_5/stroke_training.tar.gz", 1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   700,
                                                                                                   DatasetHelper.LABEL_STROKE), \
                                                                                               DatasetHelper.load_archive(
                                                                                                   "res/set_4/stroke_test.tar.gz",
                                                                                                   1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   300,
                                                                                                   DatasetHelper.LABEL_STROKE)

        training_set5 = np.row_stack((healthy_training_set5, stroke_training_set5))
        test_set5 = np.row_stack((healthy_test_set5, stroke_test_set5))

        training_labels5 = np.row_stack((healthy_training_labels5, stroke_training_labels5))
        test_labels5 = np.row_stack((healthy_test_labels5, stroke_test_labels5))

        training_set = np.row_stack((training_set4, training_set5))
        training_labels = np.row_stack((training_labels4, training_labels5))

        test_set = np.row_stack((test_set4, test_set5))
        test_labels = np.row_stack((test_labels4, test_labels5))
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set5():
        healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
            "res/set_5/healthy_training.tar.gz", 1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   700,
                                                                                                   DatasetHelper.LABEL_HEALTHY), \
                                                                                               DatasetHelper.load_archive(
                                                                                                   "res/set_4/healthy_test.tar.gz",
                                                                                                   1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   300,
                                                                                                   DatasetHelper.LABEL_HEALTHY)

        stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
            "res/set_5/stroke_training.tar.gz", 1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               700,
                                                                                               DatasetHelper.LABEL_STROKE), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_4/stroke_test.tar.gz",
                                                                                               1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               300,
                                                                                               DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))
        test_set = np.row_stack((healthy_test_set, stroke_test_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set4():
        healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
            "res/set_4/healthy_training.tar.gz", 1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   1600,
                                                                                                   DatasetHelper.LABEL_HEALTHY), \
                                                                                               DatasetHelper.load_archive(
                                                                                                   "res/set_4/healthy_test.tar.gz",
                                                                                                   1), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   400,
                                                                                                   DatasetHelper.LABEL_HEALTHY)

        stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
            "res/set_4/stroke_training.tar.gz", 1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               1600,
                                                                                               DatasetHelper.LABEL_STROKE), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_4/stroke_test.tar.gz",
                                                                                               1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               400,
                                                                                               DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))
        test_set = np.row_stack((healthy_test_set, stroke_test_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set3():
        healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
            "res/set_3/healthy_training.tar.gz", 0), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   700,
                                                                                                   DatasetHelper.LABEL_HEALTHY), \
                                                                                               DatasetHelper.load_archive(
                                                                                                   "res/set_3/healthy_test.tar.gz",
                                                                                                   0), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   300,
                                                                                                   DatasetHelper.LABEL_HEALTHY)

        stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
            "res/set_3/stroke_training.tar.gz", 0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               700,
                                                                                               DatasetHelper.LABEL_STROKE), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_3/stroke_test.tar.gz",
                                                                                               0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               300,
                                                                                               DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))
        test_set = np.row_stack((healthy_test_set, stroke_test_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set2b():
        healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_data(
            "res/set_2/healthy/training", 0), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   400,
                                                                                                   DatasetHelper.LABEL_HEALTHY), \
                                                                                               DatasetHelper.load_data(
                                                                                                   "res/set_2/healthy/test",
                                                                                                   0), \
                                                                                               DatasetHelper.generate_labels(
                                                                                                   100,
                                                                                                   DatasetHelper.LABEL_HEALTHY)

        stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_data(
            "res/set_2/stroke/training", 0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               400,
                                                                                               DatasetHelper.LABEL_STROKE), \
                                                                                           DatasetHelper.load_data(
                                                                                               "res/set_2/stroke/test",
                                                                                               0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               100,
                                                                                               DatasetHelper.LABEL_STROKE)

        training_set = np.row_stack((healthy_training_set, stroke_training_set))
        test_set = np.row_stack((healthy_test_set, stroke_test_set))

        training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
        test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))

        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set2():
        training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_2/healthy/training", 0), \
                                                               DatasetHelper.generate_labels(400,
                                                                                             DatasetHelper.LABEL_HEALTHY), \
                                                               DatasetHelper.load_data("res/set_2/healthy/test", 0), \
                                                               DatasetHelper.generate_labels(100,
                                                                                             DatasetHelper.LABEL_HEALTHY)
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def set1():
        training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_1/healthy/training", 0), \
                                                               DatasetHelper.generate_labels(700,
                                                                                             DatasetHelper.LABEL_HEALTHY), \
                                                               DatasetHelper.load_data("res/set_1/healthy/test", 0), \
                                                               DatasetHelper.generate_labels(300,
                                                                                             DatasetHelper.LABEL_HEALTHY)
        return training_set, training_labels, test_set, test_labels

    @staticmethod
    def dummy():
        training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/dummy/training", 0), \
                                                               DatasetHelper.generate_labels(30,
                                                                                             DatasetHelper.LABEL_HEALTHY), \
                                                               DatasetHelper.load_data("res/dummy/test", 0), \
                                                               DatasetHelper.generate_labels(2,
                                                                                             DatasetHelper.LABEL_HEALTHY)
        return training_set, training_labels, test_set, test_labels
