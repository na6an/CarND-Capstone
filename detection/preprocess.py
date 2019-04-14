"""
Usage:

# Data downloaded from: https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view

# Create train/test data:

python preprocess.py -i [PATH_TO_ANNOTATIONS_FILE] \
    -o [PATH_TO_ANNOTATIONS_FOLDER] \
    -r 0.2
"""

import os
import glob
import pandas as pd
import argparse
import yaml

from sklearn.model_selection import train_test_split

def to_df(inputFile):
    with open(inputFile, "r+") as f:
        data = yaml.safe_load(f)

        items = []

        for entry in data:
            filename = entry['filename']

            for a in entry['annotations']:
                items.append((filename, a['x_width'], a['y_height'], a['class'], a['xmin'], 
                            a['ymin'], a['xmin'] + a['x_width'], a['ymin'] + a['y_height']))

        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        df = pd.DataFrame(items, columns=column_name)
        
        return df


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Annonation to-CSV converter")
    parser.add_argument("-i",
                        "--inputFile",
                        help="Path to the annotation file",
                        type=str)
    parser.add_argument("-o",
                        "--outputPath",
                        help="Name of csv file output dir",
                        type=str)
    parser.add_argument("-r",
                        "--ratio",
                        help="train/test split ration",
                        type=float,
                        default=0.2)
    args = parser.parse_args()

    assert(os.path.isfile(args.inputFile))
    assert(os.path.isdir(args.outputPath))

    df = to_df(args.inputFile)

    train, test = train_test_split(df, test_size=args.ratio)

    train_labels = os.path.join(args.outputPath, 'train_labels.csv')
    test_labels = os.path.join(args.outputPath, 'test_labels.csv')
    
    train.to_csv(train_labels, index=None)
    print('Train labels saved at: {}'.format(train_labels))

    test.to_csv(test_labels, index=None)
    print('Test labels saved at: {}'.format(test_labels))

if __name__ == '__main__':
    main()