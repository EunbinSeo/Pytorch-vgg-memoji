# -*- coding: utf-8 -*-
import os

def scandir(directory, pattern):
    path = directory
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith(pattern)]

    return file_list_py