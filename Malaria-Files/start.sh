#!/bin/bash
wget -nc ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip
python3 train_test_split.py
jupyter notebook --allow-root --ip=0.0.0.0 --no-browser
