from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# processor = LlavaNextProcessor.from_pretrained("/home/ubuntu/LLaVA/checkpoints/llava-v1.6-vicuna-7b")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")



import gradio as gr
import os
import shutil


default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

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
def process(pic,q):
    # prepare image and text prompt, using the appropriate prompt template
    image = Image.open(pic)
    prompt = f"[INST] <image>\n{q} [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=3333)

    response= processor.decode(output[0], skip_special_tokens=True)
    inst_pos = response.find("[/INST]")
    return response[inst_pos + 7:]  #

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    model_status =   gr.State("llava-mistral")
    gr.Markdown(webui_title)
    with gr.Tab("llava-mistral"):
        with gr.Row():
            with gr.Column(scale=8):
                pic =  gr.Image(type="filepath")
                q = gr.Text(show_label=False, placeholder="question")
                btn = gr.Button(value="ask")
                res = gr.TextArea(show_label=False, placeholder="answer")

                btn.click(process, concurrency_limit=3,  inputs= [pic, q],  outputs=  [ res],api_name="llava")

demo.queue()
demo.launch(server_name='0.0.0.0',
         server_port=8899,share=True)
