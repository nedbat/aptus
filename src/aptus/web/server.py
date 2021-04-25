import ast
import asyncio
import base64
import functools
import io
import os
import pathlib

import PIL
import pydantic
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from aptus.compute import AptusCompute

app = FastAPI()

HERE = pathlib.Path(__file__).parent
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(HERE / "static/mainpage.html") as f:
        return f.read()

def run_in_executor(f):
    # from https://stackoverflow.com/a/53719009/14343
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))
    return inner

@run_in_executor
def compute_tile(compute):
    compute.compute_array()
    pix = compute.color_mandel()
    im = PIL.Image.fromarray(pix)
    fout = io.BytesIO()
    compute.write_image(im, fout)
    data_url = "data:image/png;base64," + base64.b64encode(fout.getvalue()).decode("ascii")
    return data_url

class ComputeSpec(pydantic.BaseModel):
    center: tuple[float, float]
    size: tuple[int, int]
    diam: tuple[float, float]
    coords: tuple[int, int, int, int]
    continuous: bool

@app.post("/tile")
async def tile(
    spec: ComputeSpec
):
    compute = AptusCompute()
    compute.center = spec.center
    compute.size = spec.size
    compute.diam = spec.diam
    compute.continuous = spec.continuous
    xmin, xmax, ymin, ymax = spec.coords

    # Reduce to a smaller tile. This needs to be moved to a function elsewhere.
    engparams = compute.engine_params()
    ss = compute.supersample
    newsize = (xmax - xmin, ymax - ymin)
    newssize = (newsize[0] * ss, newsize[1] * ss)
    compute.size = newsize
    engparams.ssize = newssize
    ri0 = engparams.ri0
    rixdx, rixdy, riydx, riydy = engparams.ridxdy
    engparams.ri0 = (
        ri0[0] + xmin * ss * rixdx + ymin * ss * rixdy,
        ri0[1] + xmin * ss * riydx + ymin * ss * riydy,
        )

    compute.create_mandel(engparams)

    data_url = await compute_tile(compute)
    return {"url": data_url}

def main():
    uvicorn.run(app, host="127.0.0.1", port=8042)
