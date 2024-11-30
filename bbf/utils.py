import os

def check_dir(path):
  if not os.path.isdir(path):
    os.makedirs(path)