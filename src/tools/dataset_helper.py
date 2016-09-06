from src.tools.simulator_data_parser import DatParser
import numpy as np
import os


class DatasetHelper:
    #
    # DIMENSIONI:
    # x_set:      ( #campioni, #feature )
    # x_labels:   ( #campioni, #classi )
    #

    def __init__(self, path):
        self.path = path
        self.dummy = 0

    def foo(self):
        print("bar")

    @staticmethod
    def load_data(path):
        """
        Loads a dataset. Reads into the path provided as parameter and returns the dataset as ndarray.
        The returned dataset dimensions are (#samples, #features)
        :param path: folder to look into
        :return: the dataset
        """

        files = [f for f in os.listdir(path)]
        number_of_files = len(files)

        number_of_views = DatParser.count_views("{}/{}".format(path, files[0]))

        first = True

        for f in files:
            filepath = "{}/{}".format(path, f)
            views_no = DatParser.count_views(filepath)
            if views_no != number_of_views:
                raise ValueError("Mismatch in number of views in {}! check .dat files!".format(path))
            if first:
                dataset = DatParser.parse_file(filepath)
                first = False
            else:
                dataset = np.row_stack((dataset, DatParser.parse_file(filepath)))

        return dataset
