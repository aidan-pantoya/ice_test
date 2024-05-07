from shapely.geometry import Polygon
import numpy
import cv2
import plotly.graph_objs as go
import math
import os
import glob
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

# Version 5/7/2024
# Author: Aidan D. Pantoya, Caleb Ebenkamp
# This script extract the ice boundaries from sea ice extent images from the national snow and ice data center

testphrase = '' # Enter a file name, or leave blank to run all

north = False

if north:
# North
    icefile = 'C:/Users/apant/Downloads/ice_test/iceimages'

else:
# South
    icefile = 'C:/Users/apant/Downloads/ice_test/iceImagesSouth'

# functions -----------------------------------------------------
def preprocess_and_correct_geometries(df): # to fix unconnected boundaries
    df['geometry'] = df['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    return df

def find_and_draw_sea_ice_boundaries(image_path,north): # this function is the main driver for the code
    img = cv2.imread(image_path)
    # cv2.imshow("img",img)

    img = img[0:int(img.shape[0]/(1.012)), :] # image cropping
    img = img[int(img.shape[0]/(15)):img.shape[0], :]

    img = img[:,0:int(img.shape[1]/(1.28))]
    img = img[:,int(img.shape[1]/35):img.shape[1]]

    img = cv2.blur(img,(3,3)) # blur

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
    _, thresholded = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY) # threshold to binary
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours
    
    img_height, img_width = img.shape[:2]
    center_x, center_y = img_width / 2, img_height / 2
    max_radius = min(center_x, center_y)
    geo_boundaries = []
    for contour in contours:
        if cv2.contourArea(contour) < 5: # ignore tiny contours
            continue
        cv2.drawContours(img, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
        geo_contour = []
        if north: # different north and south boundary conversions
            scale_factor = 0.41
            for point in contour:
                x, y = point[0]
                dx, dy = x - center_x, y - center_y
                radius = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)
                lat = 90 - (radius / max_radius * 90 * scale_factor)
                lon = (angle / math.pi * 180) % 360
                lon = -lon
                lon +=44.8
                if lon > 180:
                    lon -= 360

                lat = max(min(lat, 90), -90)
                geo_contour.append((lon, lat))
        else:
            scale_factor = 0.4
            for point in contour:
                x, y = point[0]
                dx, dy = x - center_x, y - center_y
                radius = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)
                lat = (radius / max_radius * 90 * scale_factor) - 90
                lon = (angle / math.pi * 180) % 360
                lon += 90
                if lon > 180:
                    lon -= 360

                lat = max(min(lat, 90), -90)
                geo_contour.append((lon, lat))

        geo_boundaries.append(geo_contour)
    # cv2.imshow('og', img)
    print(image_path)

    # cv2.imwrite(image_path+'_processed.png',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return geo_boundaries

def plot_boundaries_on_map(boundaries): # put extracted boundaries on globe graph
    fig = go.Figure()
    for boundary in boundaries:
        lons, lats = zip(*boundary)
        fig.add_trace(go.Scattergeo(
            lon = lons,
            lat = lats,
            mode = 'lines',
            line = dict(width = 2, color = 'blue'),
        ))
    fig.update_layout(
        title = 'Sea Ice Boundaries',
        geo = dict(
            projection_type = 'orthographic',
            projection_rotation = dict(lon = 0, lat = 90, roll = 0),
            showland = True,
            landcolor = "LightGreen",
            showocean = True,
            oceancolor = "LightBlue"
        ),
    )
    fig.show()

def find_png_files(directory, pattern): # find all images in directory with given pattern
    search_pattern = os.path.join(directory, f'*{pattern}*.png')
    matching_files = glob.glob(search_pattern)
    return matching_files

def create_geometry_df(polygons): # combine boundaries
    merged_polygon = unary_union(polygons)
    df = pd.DataFrame({'geometry': [merged_polygon]})
    return df

def extract_coordinates(geometry): # get converted boundaries
    if isinstance(geometry, MultiPolygon):
        return [list(polygon.exterior.coords) for polygon in geometry.geoms]
    elif isinstance(geometry, Polygon):
        return [list(geometry.exterior.coords)]
    else:
        raise TypeError("Geometry must be a Polygon or MultiPolygon")

# end of functions -----------------------------------------------------------

foundicefiles = find_png_files(icefile,testphrase)

geo_boundaries = []
for foundicefile in foundicefiles: # goes through each image
    boundaries = find_and_draw_sea_ice_boundaries(foundicefile,north)
    for coords in boundaries:
        df = create_geometry_df([Polygon(coords)])
        if df is not None:
            geo_boundaries.append(df)

extracted_boundaries = []
for df in geo_boundaries: # gets all boundaries
    for _, row in df.iterrows():
        coords_list = extract_coordinates(row['geometry'])
        for coords in coords_list:
            extracted_boundaries.append(coords)

plot_boundaries_on_map(extracted_boundaries) # plots all boundaries
