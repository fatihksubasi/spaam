#!/usr/bin/python

import argparse
import csv
from spaam import SPAAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Provide file path to csv file containing Pixel and corresponding 3D Coordinate Values')
    parser.add_argument("filePath", type=str, help='path to relevant csv file')
    args = parser.parse_args()

    with open(args.filePath, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        pImage = []
        pWorld = []
        try:
            for row in rows:
                pImage.append([float(row['\ufeffDisplay Horizontal Pixel']),
                               float(row['Display Vertical Pixel'])])
                pWorld.append([float(row['HMD Position X']),
                               float(row['HMD Position Y']), float(row['HMD Position Z'])])

        except csv.Error as e:
            print(e)

    spaam = SPAAM(pImage, pWorld)
    G = spaam.get_camera_matrix()

    print("G Matrix:")
    print(G)
