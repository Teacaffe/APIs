from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageEnhance
from io import BytesIO
import math
import requests

app = FastAPI()


class StaticMapParams(BaseModel):
    coordinates: str
    map_type: str = "roadmap"
    contrast: int = 1
    greyscale: bool = False
    draw_rectangle: bool = False
    rectangle_rgba: str = "ff0000ff"
    rectangle_weight: int = 1


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
               f"&key=AIzaSyAN4RaYNeTo-BXcPBKG_gsjNgSW4FHmYGs")
    elif provider.lower() == "geoapify":
        rectangle = f"&geometry=rect:{minLon},{minLat},{maxLon},{maxLat};linewidth:{rectangle_weight};linecolor:%23{rectangle_rgba[0:6]};lineopacity:{int(rectangle_rgba[6:8], 16) / 255}" if draw_rectangle else ""
        url = (f"https://maps.geoapify.com/v1/staticmap?"
               f"style={map_type}"
               f"&center=lonlat:{center_lon},{center_lat}"
               f"&zoom={zoom}"
               f"&width={width * 2}&height={height * 2}"
               f"&scaleFactor=2"
               f"{rectangle}"
               f"&apiKey=c8e849852f404fac9bc96a97e10447a2")
    print(f"Map url: {url}")
    response = requests.get(url)
    output = Image.open(BytesIO(response.content))
    output = output if not greyscale else output.convert('L')
    output = output if contrast == 1 else ImageEnhance.Contrast(
        output.convert("RGB")).enhance(contrast)
    outputIO = BytesIO()
    output.save(outputIO, format="PNG")
    outputIO.seek(0)
    return outputIO


@app.get("/static_map")
def new_map(params: StaticMapParams = Depends()):
    coordinates = [float(x) for x in params.coordinates.split(",")]
    if len(coordinates) != 4:
        raise HTTPException(
            status_code=400, detail="Coordinates parameter requires four [float] values: minLat, minLon, maxLat, maxLon")
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
