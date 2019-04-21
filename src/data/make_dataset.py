# -*- coding: utf-8 -*-
######################## Machine Learning - Code ###############################
# Author: Jimena Baripatti
# Email: jimenabaripatti@gmail.com
###########################################################################


import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import os
from sklearn.model_selection import train_test_split





@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

def make_dataset(input_filepath, output_filepath):
    with open(input_filepath, 'r') as f:
        first_line = f.readline().strip()
    f.close()

    num_sig = int(first_line.split(" ")[0])
    num_bg = int(first_line.split(" ")[1])

    y = np.append(np.ones(num_sig), np.zeros(num_bg))
    X = np.loadtxt(input_filepath, skiprows=1)
    data = np.append(X, np.reshape(y, [y.shape[0], 1]), axis=1)

    np.savetxt(output_filepath, data, delimiter= ',')

    return data
def split_data(dataset):
    #split dataset
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    # split attributes and label
    X_train = train_set[:,0:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,0:-1]
    y_test = test_set[:,-1]

    return X_train, y_train, X_test, y_test








if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


