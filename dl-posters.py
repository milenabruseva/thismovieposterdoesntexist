import multiprocessing
import os
import pandas as pd
import json
import requests
from multiprocessing.pool import ThreadPool

# Input: genre(s), max. number of movies, sorted by vote_count or popularity, folder to save images in, resolution/size

# Load dataset table from .pkl

# view only entries with the given genres (logic AND)

# sort for vote_count or popularity

# Create list of with tuples of IDs and URL

# Start download via ThreadPool

# https://likegeeks.com/downloading-files-using-python/#Download_multiple_files_Parallelbulk_download