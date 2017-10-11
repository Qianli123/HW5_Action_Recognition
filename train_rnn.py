import tensorflow as tf

import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import h5py

data_directory = ''
class_list, train, test = getUCF101(base_directory = data_directory)
