from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageEnhance
from io import BytesIO
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geopy.distance import geodesic
import numpy as np
import math
import requests
import os

app = FastAPI()


class StaticMapParams(BaseModel):
    bounds: str
    map_type: str = "roadmap"
    contrast: int = 1
    greyscale: bool = False
    draw_rectangle: bool = False
    rectangle_rgba: str = "ff0000ff"
    rectangle_weight: int = 1


def round_half_up(n: float, decimals: int = 0) -> float:
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def make_elevation_matrix(min_lat, min_lon, max_lat, max_lon, lat_size, lon_size, progress_callback=None):
    # Progress callback format: def progress_callback(current, max)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    api_key = os.getenv("GOOGLE_MAPS_PLATFORM_API_KEY")
    elevation_matrix = np.zeros((lat_size, lon_size), dtype=float)
    lat_step = (max_lat - min_lat) / (lat_size - 1)
    lon_step = (max_lon - min_lon) / (lon_size - 1)
    batch_size = 512
    start_index = 0
    i = start_index
    matrix_area = lat_size * lon_size
    min_elevation = None
    max_elevation = None
    if progress_callback != None:
        progress_callback(0, matrix_area)
    while True:
        end_index = start_index + (batch_size - 1)
        end_index = end_index if end_index < matrix_area else matrix_area - 1
        batch_locations_list = []
        i = start_index
        while i <= end_index:
            row = math.floor(i / lon_size)
            col = i % lon_size
            new_lat = round_half_up(max_lat - (lat_step * row), 6)
            new_lon = round_half_up(min_lon + (lon_step * col), 6)
            batch_locations_list.append(f"{new_lat},{new_lon}")
            i += 1
        batch_locations_param = "|".join(batch_locations_list)
        batch_params = {
            'locations': batch_locations_param,
            'key': api_key
        }
        batch_data = requests.get(url, params=batch_params).json()
        batch_elevations = []
        if batch_data['status'] != 'OK':
            print(f"Error: {batch_data['status']}")
            break
        batch_elevations = [result['elevation']
                            for result in batch_data['results']]
        min_batch_elevation = min(batch_elevations)
        max_batch_elevation = max(batch_elevations)
        min_elevation = min_batch_elevation if min_elevation == None or min_batch_elevation < min_elevation else min_elevation
        max_elevation = max_batch_elevation if max_elevation == None or max_batch_elevation > max_elevation else max_elevation
        i = start_index
        while i <= end_index:
            row = math.floor(i / lon_size)
            col = i % lon_size
            elevation_matrix[row][col] = batch_elevations[i - start_index]
            i += 1
        if progress_callback != None:
            progress_callback(end_index + 1, matrix_area)
        start_index = end_index + 1
        if start_index >= matrix_area:
            break
    return (np.array(elevation_matrix), min_elevation, max_elevation)


def generate_static_map(minLat, minLon, maxLat, maxLon,
                        map_type="roadmap",
                        contrast=1,
                        greyscale=False,
                        draw_rectangle=False,
                        rectangle_rgba="ff0000ff",
                        rectangle_weight=1):
    center_lat = (minLat + maxLat) / 2
    center_lon = (minLon + maxLon) / 2

    def lat_to_pixel(lat, zoom):
        WORLD_SIZE = 256 * (2 ** zoom)
        return math.ceil((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * WORLD_SIZE)

    def lon_to_pixel(lon, zoom):
        WORLD_SIZE = 256 * (2 ** zoom)
        return math.ceil(((lon + 180) / 360) * WORLD_SIZE)

    def calculate_zoom_level():
        ZOOM_MAX = 21
        zoom = ZOOM_MAX
        while zoom > 0:
            lat_pixel_diff = abs(lat_to_pixel(
                maxLat, zoom) - lat_to_pixel(minLat, zoom))
            lon_pixel_diff = abs(lon_to_pixel(
                maxLon, zoom) - lon_to_pixel(minLon, zoom))
            if lat_pixel_diff < 640 and lon_pixel_diff < 640:
                break
            zoom -= 1
        return zoom

    def round_half_up(n: float, decimals: int = 0) -> float:
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier
    zoom = calculate_zoom_level()
    min_x_pixel = lon_to_pixel(minLon, zoom)
    max_x_pixel = lon_to_pixel(maxLon, zoom)
    width = math.ceil(abs(max_x_pixel - min_x_pixel))
    min_y_pixel = lat_to_pixel(minLat, zoom)
    max_y_pixel = lat_to_pixel(maxLat, zoom)
    height = math.ceil(abs(max_y_pixel - min_y_pixel))
    width = min(width, 640)
    height = min(height, 640)
    url = ""

    google_map_types = ("roadmap", "satellite", "terrain", "hybrid")
    geoapify_map_types = ("osm-carto", "osm-bright", "osm-bright-grey", "osm-bright-smooth", "klokantech-basic", "osm-liberty", "maptiler-3d", "toner", "toner-grey", "positron",
                          "positron-blue", "positron-red", "dark-matter", "dark-matter-brown", "dark-matter-dark-grey", "dark-matter-dark-purple", "dark-matter-purple-roads", "dark-matter-yellow-roads")
    map_type = map_type.lower() if map_type.lower(
    ) in google_map_types or map_type.lower() in geoapify_map_types else "roadmap"
    provider = "geoapify" if map_type.lower() in geoapify_map_types else "google"

    rectangle_rgba = rectangle_rgba if rectangle_rgba[0:
                                                      2] != "0x" else rectangle_rgba[2:]
    rectangle_rgba = f"{rectangle_rgba}ff" if len(
        rectangle_rgba) == 6 else rectangle_rgba
    center_lat = round_half_up(center_lat, 6)
    center_lon = round_half_up(center_lon, 6)

    if provider.lower() == "google":
        rectangle = (
            f"&path=color:0x{rectangle_rgba}|weight:{rectangle_weight}|{minLat},{minLon}|{minLat},{maxLon}|{maxLat},{maxLon}|{maxLat},{minLon}|{minLat},{minLon}" if draw_rectangle else "")
        url = (f"https://maps.googleapis.com/maps/api/staticmap?"
               f"maptype={map_type}"
               f"&center={center_lat},{center_lon}"
               f"&zoom={zoom}"
               f"&size={width}x{height}"
               f"&scale=2"
               f"{rectangle}"
               f"&key={os.getenv('GOOGLE_MAPS_PLATFORM_API_KEY')}")
    elif provider.lower() == "geoapify":
        rectangle = f"&geometry=rect:{minLon},{minLat},{maxLon},{maxLat};linewidth:{rectangle_weight};linecolor:%23{rectangle_rgba[0:6]};lineopacity:{int(rectangle_rgba[6:8], 16) / 255}" if draw_rectangle else ""
        url = (f"https://maps.geoapify.com/v1/staticmap?"
               f"style={map_type}"
               f"&center=lonlat:{center_lon},{center_lat}"
               f"&zoom={zoom}"
               f"&width={width * 2}&height={height * 2}"
               f"&scaleFactor=2"
               f"{rectangle}"
               f"&apiKey={os.getenv('GEOAPIFY_API_KEY')}")
    # print(f"Map url: {url}")
    response = requests.get(url)
    output = Image.open(BytesIO(response.content))
    output = output if not greyscale else output.convert('L')
    output = output if contrast == 1 else ImageEnhance.Contrast(
        output.convert("RGB")).enhance(contrast)
    outputIO = BytesIO()
    output.save(outputIO, format="PNG")
    outputIO.seek(0)
    return outputIO


def generate_elevation_contour(minLat, minLon, maxLat, maxLon, vertical_size, horizontal_size):
    def print_progress(current, max):
        progress = current / max
        print(f"Progress: {current}/{max} - {progress:.2%}")
    elevation_matrix, min_elev, max_elev = make_elevation_matrix(
        minLat, minLon, maxLat, maxLon, vertical_size, horizontal_size, progress_callback=print_progress)
    elev_fig, elev_ax = plt.subplots(figsize=(horizontal_size, vertical_size))
    elev_contour = elev_ax.contourf(
        elevation_matrix, origin='upper', cmap='terrain')
    elev_ax.set_xticks([])
    elev_ax.set_yticks([])
    # elev_image_buffer = BytesIO()
    # plt.savefig(elev_image_buffer, format="png", dpi=300, bbox_inches="tight")
    # plt.savefig("Plots/elevation_contour_clear.png", dpi=300, bbox_inches="tight")
    elev_cax = make_axes_locatable(elev_ax).append_axes(
        "right", size="5%", pad=0.1)
    elev_cbar = plt.colorbar(elev_contour, cax=elev_cax)
    elev_cbar.set_label("Elevation (meters)")
    elev_ax.set_title("Elevation contour")
    # plt.savefig("Plots/elevation_contour.png", dpi=300, bbox_inches="tight")
    elev_contour_io = BytesIO()
    plt.savefig(elev_contour_io, format="png", dpi=300, bbox_inches="tight")
    plt.close(elev_fig)
    return elev_contour_io


@app.get("/static_map")
def new_map(params: StaticMapParams = Depends()):
    coordinates = [float(x) for x in params.bounds.split(",")]
    if len(coordinates) != 4:
        raise HTTPException(
            status_code=400, detail="Bounds parameter requires four [float] values: minLat, minLon, maxLat, maxLon")
    minLat = coordinates[0]
    minLon = coordinates[1]
    maxLat = coordinates[2]
    maxLon = coordinates[3]
    mapIO = generate_static_map(minLat, minLon, maxLat, maxLon,
                                params.map_type,
                                params.contrast,
                                params.greyscale,
                                params.draw_rectangle,
                                params.rectangle_rgba,
                                params.rectangle_weight)
    return StreamingResponse(mapIO, media_type="image/png")


@app.get("/elevation_contour")
def new_elev_contour_map(bounds: str, horizontal_resolution: int = None, vertical_resolution: int = None):
    coordinates = [float(x) for x in bounds.split(",")]
    if len(coordinates) != 4:
        raise HTTPException(
            status_code=400, detail="Bounds parameter requires four [float] values: minLat, minLon, maxLat, maxLon")
    minLat = coordinates[0]
    minLon = coordinates[1]
    maxLat = coordinates[2]
    maxLon = coordinates[3]
    # contourIO = generate_elevation_contour(minLat, minLon, maxLat, maxLon, vertical_size, horizontal_size)

    def print_progress(current, max):
        progress = current / max
        print(f"Progress: {current}/{max} - {progress:.2%}")
    width = geodesic((minLat, minLon), (minLat, maxLon)).meters
    height = geodesic((minLat, minLon), (maxLat, minLon)).meters
    aspect_ratio = height / width
    print(f"Size: {width}x{height} --> Aspect ratio: {aspect_ratio}")
    horizontal_resolution = horizontal_resolution if horizontal_resolution != None else 100
    vertical_resolution = vertical_resolution if vertical_resolution != None else math.ceil(
        horizontal_resolution * aspect_ratio)
    elevation_matrix, min_elev, max_elev = make_elevation_matrix(
        minLat, minLon, maxLat, maxLon, vertical_resolution, horizontal_resolution, progress_callback=print_progress)
    figWidth = 10
    figHeight = figWidth * aspect_ratio
    print(
        f"figSize: {figWidth}x{figHeight} --> Aspect ratio: {figHeight / figWidth}")
    elev_fig, elev_ax = plt.subplots(figsize=(figWidth, figHeight))
    elev_contour = elev_ax.contourf(
        elevation_matrix, origin='upper', cmap='terrain')
    elev_ax.set_xticks([])
    elev_ax.set_yticks([])
    # elev_image_buffer = BytesIO()
    # plt.savefig(elev_image_buffer, format="png", dpi=300, bbox_inches="tight")
    # plt.savefig("Plots/elevation_contour_clear.png", dpi=300, bbox_inches="tight")
    elev_cax = make_axes_locatable(elev_ax).append_axes(
        "right", size="5%", pad=0.1)
    elev_cbar = plt.colorbar(elev_contour, cax=elev_cax)
    elev_cbar.set_label("Elevation (meters)")
    elev_ax.set_title("Elevation contour")
    # plt.savefig("Plots/elevation_contour.png", dpi=300, bbox_inches="tight")
    elev_contour_io = BytesIO()
    plt.savefig(elev_contour_io, format="png", dpi=300, bbox_inches="tight")
    plt.close(elev_fig)
    elev_contour_io.seek(0)
    return StreamingResponse(elev_contour_io, media_type="image/png")
