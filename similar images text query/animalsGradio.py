# -*- coding: utf-8 -*-
import animals2Oracle
from fileinput import filename
from PIL import Image
import requests
# import gradio
import gradio as gr
import os
import shutil

import torch
import numpy as np
from PIL import Image as imim
import csv
from glob import glob
from pathlib import Path
import os
from statistics import mean


# import oracle_ai_vector_search as ovs


flag_csv_logger = gr.CSVLogger()

# for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
#                                                       streaming=streaming):
#     resp = answer_result.llm_output["answer"]
#     history = answer_result.history
#     history[-1][-1] = resp
#     yield history, ""

# logger.info(
#     f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
# flag_csv_logger.flag([query, vs_path, history, mode],
#                      username=FLAG_USER_NAME)


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """SEHUB JS
"""
init_message = f"""欢迎使用    
"""

# from transformers import AutoModelForCausalLM, LlamaTokenizer
# from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(
#         'THUDM/cogvlm-chat-hf',
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
# device_map = infer_auto_device_map(model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
# model = load_checkpoint_and_dispatch(
#     model,
#     '/home/opc/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c',   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
#     device_map=device_map,
# )
# model = model.eval()

# check device for weights if u want to
# for n, p in model.named_parameters():
#     print(f"{n}: {p.device}")
resultDocs=[]

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)


# def clip2Vec(img_path):
#     print(11, img_path)
#     image = preprocess(imim.open(img_path)).unsqueeze(0).to(device)
#     image_features = model.encode_image(image)
#     # ndarray_data = np.asarray(image_features.cpu())
#     ndarr = image_features.cpu().detach().numpy()
#     print('22', ndarr.squeeze().shape)
#     ll = ndarr.tolist()
#     print(ll)
#     return ll[0]

import glmHelper
def mirror(x):
    return x
def identity(x, state):
    state += 1
    return x, state, state
def get_filename_from_path(path):
    return os.path.basename(path)
from riva_stream_file import file2sum, file2sumCN

def asr(file,audioLang):
    print(file)
    # filename=file.name
    if 'English' == audioLang:

      text = file2sum(file)
    else:
      text = file2sumCN(file)
        
    return text


def update_slider(val):
    # Update the slider value to the new value obtained from input
    return val

import json
with gr.Blocks() as demo:
    with gr.Tab("Search Similar Images by using Text Query"):
        with gr.Row():
            with gr.Column(scale=8):
                audioLang =  gr.Radio(["English", "普通话"],value='普通话', label="语音识别的语言")

                audio = gr.Audio(label='recording / 录音', scale=5, sources=["microphone"], show_download_button=True,
                             type='filepath')
                # pic = gr.Image(type="filepath", height=444)
                t1 = gr.Text(label='search text(AI auto select)/搜索')
                # ttype = gr.Text(label='search type')
                ttype =  gr.Radio(["action", "breed", "env"],value='action', label="search type")
                slider = gr.Slider(minimum= 1,maximum= 15,interactive=True, step=1, value=4, info="limit to search"),

                s_btn = gr.Button("Search/向量查询")
                checkBtn = gr.Button("AI check/大模型检查")

                # limit = gr.Number(2, interactive=True,
                #                   label="the number of similar images to show")
            with gr.Column(scale=8):
                outputs = gr.Gallery(label="Result Similar Images")
        # slider[0].release(mirror, inputs=[slider[0]], outputs=[t1], api_name="predict")
        audio.change(asr, [audio,audioLang], t1)
        @s_btn.click(inputs=[t1, ttype,  slider[0]], outputs=[outputs,ttype])
        def textSearch(query, ttype , limit):
            query= glmHelper.toEng(query)
            AiInfer= glmHelper.smartIntent(query)
            AiInfer = AiInfer.lower()
            if "a. animal action" in AiInfer:
                 ttype='action'
            elif "b. animal species" in AiInfer:
                 ttype='breed'
            elif "c. environment" in AiInfer:
                 ttype='env'
                
            print('the search type is '+ ttype)
            # update_slider()
            docs = animals2Oracle.searchVecs(query,ttype , limit)
            global resultDocs
            resultDocs = docs
            arr = []
            for doc in docs:
                # arr.append((doc[0].metadata['img_src'],  doc[0].page_content ))
                descriptionText  = glmHelper.toLocale(str(doc.get('typeSearch')))
                arr.append((doc.get('path'), descriptionText ))

            return [arr,ttype]
        @checkBtn.click(inputs=[t1], outputs=outputs)
        def check2(query):
            query= glmHelper.toEng(query)
            arr = []
            for doc in resultDocs:
                # arr.append((doc[0].metadata['img_src'],  doc[0].page_content ))
                if 'yes, relevant' in glmHelper.helpcheck(doc.get('typeSearch'),query).lower() :
                    descriptionText  = glmHelper.toLocale(str(doc.get('typeSearch')))
                    arr.append((doc.get('path'),  descriptionText ))

            return arr
demo.queue()
demo.launch(server_name='0.0.0.0',
            server_port=8899, share=False,  show_api=True)
# (demo
#  .queue(concurrency_count=3)
#  .launch(server_name='0.0.0.0',
#          server_port=7860,
#          show_api=True,
#          share=False,
#          inbrowser=True))
