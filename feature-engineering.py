import os
from collections import namedtuple, OrderedDict

import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - {%(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d}",
)
logger = logging.getLogger(__name__)


# here we start the feature engineering
# train and test data will undergo the same transformations
# we will use the same variable names for both datasets
def feature_engineering(df):
    cal_cities = pd.read_csv("cal_cities_lat_long.csv")
    cal_pops_cities = pd.read_csv("cal_populations_city.csv")

    df.loc[
        df["ocean_proximity"] == "<1H OCEAN", "ocean_proximity"
    ] = "WITHIN HOUR TO OCEAN"
    df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    df.drop(["total_rooms"], axis=1, inplace=True)
    df.drop(["total_bedrooms"], axis=1, inplace=True)
    df.drop(["households"], axis=1, inplace=True)

    file_path = os.path.join("CA_Counties", "CA_Counties_TIGER2016.shp")
    cali_shp = gpd.read_file(file_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=cali_shp.crs,
    )
    gdf = gdf.set_crs(cali_shp.crs)

    file_path = os.path.join("CA_Counties", "CA_Counties_TIGER2016.shp")
    cali_shp = gpd.read_file(file_path)
    cali_shp = cali_shp.to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=cali_shp.crs,
    )
    gdf = gdf.to_crs("EPSG:4326")

    CityTuple = namedtuple(
        "City",
        [
            "Name",
            "Latitude",
            "Longitude",
            "pop_april_1980",
            "pop_april_1990",
            "pop_april_2000",
            "pop_april_2010",
        ],
    )
    city_map = dict()
    for _, row in cal_cities.iterrows():
        city_map[row["Name"]] = CityTuple(
            row["Name"], row["Latitude"], row["Longitude"], 0, 0, 0, 0
        )

    for _, row in cal_pops_cities.iterrows():
        if row["City"] in city_map:
            tuple_ = city_map[row["City"]]
            tuple_ = tuple_._replace(
                pop_april_1980=row["pop_april_1980"],
                pop_april_1990=row["pop_april_1990"],
                pop_april_2000=row["pop_april_2000"],
                pop_april_2010=row["pop_april_2010"],
            )
            city_map[row["City"]] = tuple_

    tuple_list = [tuple_ for tuple_ in city_map.values()]
    gdf_cities = gpd.GeoDataFrame(
        tuple_list,
        geometry=gpd.points_from_xy(
            [tuple_[2] for tuple_ in tuple_list], [tuple_[1] for tuple_ in tuple_list]
        ),
        crs=cali_shp.crs,
    )
    gdf_cities = gdf_cities.to_crs("EPSG:4326")

    gdf_cities[
        ["pop_april_1980", "pop_april_1990", "pop_april_2000", "pop_april_2010"]
    ] = gdf_cities[
        ["pop_april_1980", "pop_april_1990", "pop_april_2000", "pop_april_2010"]
    ].astype(
        float
    )
    gdf_cities["Large_City"] = gdf_cities["pop_april_2010"] > 250000
    large_cities = gdf_cities[gdf_cities["Large_City"] == True]

    large_cities_lat_lon = large_cities[["Name", "Latitude", "Longitude"]]
    large_cities_dictionaries = large_cities_lat_lon.to_dict(
        "records", into=OrderedDict
    )

    for large_city in large_cities_dictionaries:
        large_city_base_name = large_city["Name"]
        large_city_base_name = large_city_base_name.replace(" ", "_").lower()
        large_city_enriched_name = large_city_base_name + "_km_distance"
        gdf[large_city_enriched_name] = gdf.apply(
            lambda row: geodesic(
                (row["latitude"], row["longitude"]),
                (large_city["Latitude"], large_city["Longitude"]),
            ).kilometers,
            axis=1,
        )

    distance_cols = [col for col in gdf.columns if col.endswith("_km_distance")]
    distance_data = gdf[distance_cols]
    distance_data.head()
    gdf["min_distance"] = distance_data.min(axis=1)
    gdf["max_distance"] = distance_data.max(axis=1)
    gdf["min_distance_km_col"] = distance_data.idxmin(axis=1)
    gdf["max_distance_km_col"] = distance_data.idxmax(axis=1)
    gdf.drop(columns=distance_cols, inplace=True)

    gdf["min_distance_city_name"] = gdf["min_distance_km_col"].apply(
        lambda x: x.split("_km_distance")[0].replace("_", " ")
    )
    gdf.drop(
        columns=["min_distance_km_col", "max_distance_km_col", "max_distance"],
        inplace=True,
    )
    gdf.drop(columns=["latitude", "longitude"], inplace=True)

    large_cities = large_cities.rename(columns={"Name": "city_name_pops"})
    large_cities["city_name_pops"] = large_cities["city_name_pops"].apply(
        lambda x: x.lower()
    )
    large_cities["city_name_pops"] = large_cities["city_name_pops"].apply(
        lambda x: x.replace(" ", "_")
    )

    gdf["min_distance_city_name"] = gdf["min_distance_city_name"].apply(
        lambda x: x.replace(" ", "_")
    )

    merged_gdf = gdf.merge(
        large_cities,
        how="left",
        left_on="min_distance_city_name",
        right_on="city_name_pops",
    )
    merged_gdf.drop(
        columns=["city_name_pops", "Latitude", "Longitude", "geometry_y", "Large_City"],
        inplace=True,
    )
    merged_gdf.rename(columns={"geometry_x": "geometry"}, inplace=True)
    merged_gdf.drop(columns=["geometry"], inplace=True)
    merged_gdf.drop(columns=["min_distance_city_name"], inplace=True)

    merged_gdf[
        ["pop_april_1980", "pop_april_1990", "pop_april_2000", "pop_april_2010"]
    ] = merged_gdf[
        ["pop_april_1980", "pop_april_1990", "pop_april_2000", "pop_april_2010"]
    ].apply(
        np.log
    )

    merged_gdf.drop(columns=["pop_april_1980", "pop_april_2000", "pop_april_2010"], inplace=True)


    return merged_gdf


if __name__ == "__main__":
    # Load
    logger.info("Loading data")
    df = pd.read_csv("housing.csv")
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Clean
    logger.info("Feature engineering")

    # Train
    logger.info("Training model")

    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    num_processor = Pipeline([("std_scaler", StandardScaler())])

    cat_processor = Pipeline(
        [("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                num_processor,
                [
                    "housing_median_age",
                    "population",
                    "median_income",
                    "rooms_per_household",
                    "bedrooms_per_room",
                    "bedrooms_per_household",
                    "population_per_household",
                    "min_distance",
                    # "pop_april_1980",
                    "pop_april_1990",
                    # "pop_april_2000",
                    # "pop_april_2010",
                ],
            ),
            # ("cat", cat_processor, ["min_distance_city_name", "ocean_proximity"]),
            ("cat", cat_processor, ["ocean_proximity"]),
        ]
    )

    # shape
    logger.info("X_train shape")
    logger.info(X_train.shape)
    logger.info("X_test shape")
    logger.info(X_test.shape)
    # columns
    logger.info("X_train columns")
    logger.info(list(X_train.columns))
    logger.info("X_test columns")
    logger.info(list(X_test.columns))

    for model in [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE for {model.__class__.__name__}: {rmse}")
