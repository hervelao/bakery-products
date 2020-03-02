import os
import pandas as pd

class Bimbo():

    def get_data(self):
        """
        This function returns all Bimbo data sets
        as DataFrames within a Python dict.
        """
        data_dict = {}
        abs_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        for file in os.listdir('{}/data/csv'.format(abs_path)):
            data_dict[file[:-4]] = pd.read_csv('{}/data/csv/{}'.format(abs_path, file))
        return data_dict

if __name__ == '__main__':
    bimbo = Bimbo()
    print(olist.get_data())
