import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, PULP_CBC_CMD

def main():
    shape_file_path = input('Please input shapefile path name: ')


if __name__ == "__main__":
    main()