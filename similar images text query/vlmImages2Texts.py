# pip install sentence-transformers
# pip install langchain


from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from gradio_client import Client, file
import time
import concurrent
import array
from statistics import mean
import os
from pathlib import Path
from glob import glob
import csv
from PIL import Image as imim
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import asyncio
import contextlib
import enum
import json
import logging
import uuid
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from datetime import datetime
import numpy as np



def get_jpg_files(directory):
    jpg_files = []  # 创建一个空列表来存储.jpg文件的路径

    # 使用os.walk遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 使用os.path.splitext检查文件的扩展名
            if os.path.splitext(file)[1] == '.jpg':
                # 用os.path.join把目录和文件名合并为一个完整的路径
                jpg_files.append(os.path.join(root, file))

    return jpg_files


def v2t1(img):

    result = client.predict(
        pic=file(img),
        q="what are the animal categories in the picture? less than 100  words. ",
        api_name="/llava"
    )

    print(result)
    return result

def v2t2(img):

    result = client.predict(
        pic=file(img),
        q="What is the environment?  less than 100 words.",
        api_name="/llava"
    )

    print(result)
    return result
def v2t3(img):

    result = client.predict(
        pic=file(img),
        q="What are these animals doing right now?  less than 100 words.",
        api_name="/llava"
    )

    print(result)
    return result

def get_folders_and_images(root_dir):
    global count
    for folder_name in os.listdir(root_dir):
        ## reduce some images 
       
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            slasarr = folder_path.split('/')
            class1 = slasarr[-1]
            print("Folder:", class1)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print("Image:", image_path)
                    count = count+1
                    with open(image_path+'.kind', "w") as f:
                          f.write(v2t1(image_path))
                    with open(image_path+'.env', "w") as f:
                          f.write(v2t2(image_path))
                    with open(image_path+'.action', "w") as f:
                          f.write(v2t3(image_path))
                    with open("summary_animals.csv", "a") as file:
    # Append the data to the file
                        file.write(f"{image_path},{image_path}.kind,{image_path}.env,{image_path}.action\n")
                        if count%15==0 and count!=0:
                             break
                    # metadata = {"id": str(uuid.uuid1()), "image_path": image_path, 'description':description, }
                    # doc_langchain = Document(page_content=description, metadata=metadata)
                    # # documents_langchain.append(doc_langchain)
                    # vector_store_max = OracleVS.from_documents(
                    #     [doc_langchain],
                    #     embedding_function,
                    #     client=connection,
                    #     table_name="Documents_EricCOSINE",
                    #     distance_strategy=DistanceStrategy.COSINE,
                    # )

### this is the url of llava model, refer to llavaGradio.py
client = Client("http://127.1.1.1:8899/")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
if __name__ == "__main__":
    # tableName = 'ericAnimals'
    # documents_langchain = []

    # for doc in documents_json_list:
    #     metadata = {"id": doc["id"], "path": doc["link"], ''}
    #     doc_langchain = Document(page_content=doc["text"], metadata=metadata)
    #     documents_langchain.append(doc_langchain)
        
   
    directory = './animal_data/'

    count = 0
    start = time.perf_counter()
    get_folders_and_images(directory)
    end = time.perf_counter()
    print('The image to text process has taken  ',
          end - start, 'seconds')
    print('Total images count: ', count)
