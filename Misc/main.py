import numpy as np
from fastapi import FastAPI

app = FastAPI()

my_list = [1,2,3,4]

# App decorator that runs the function when the root URL is accessed
@app.get("/")
def root():
    return "Hello World" 

@app.get("/list{index}")
def get_list(index: int):
    return { "item": my_list[index] }

# /search_list?item=3, using the query parameter "item" to search for a value in the list
@app.get("/search_list")
def search_list(item: int):
    if item in my_list:
        return { "found": True, "index": my_list.index(item) }
    else:
        return False

# ERROR:    Error loading ASGI app. Could not import module "Misc\main".
# This happens because Uvicorn expects a Python module path, not a file path.

# Run the server with the correct module path:
# uvicorn Misc.main:app --reload