'''
Author: Egrt
Date: 2022-01-13 13:34:10
LastEditors: [egrt]
LastEditTime: 2022-08-15 19:40:32
FilePath: \MaskGAN\app.py
'''
from HEAT import HEAT
import gradio as gr
import os
heat = HEAT()

# --------模型推理---------- #
def inference(img):
    image_result = heat.detect_one_image(img)
    return image_result

# --------网页信息---------- #  
title = "HEAT"
description = "HEAT: Holistic Edge Attention Transformer for Structured Reconstruction   @Luuuu"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2108.10257' target='_blank'>HEAT: Holistic Edge Attention Transformer for Structured Reconstruction </a> | <a href='https://github.com/JingyunLiang/SwinIR' target='_blank'>Github Repo</a></p>"
example_img_dir  = 'images/'
example_img_name = os.listdir(example_img_dir)
examples=[[os.path.join(example_img_dir, image_path)] for image_path in example_img_name if image_path.endswith(('.jpg','.jpeg', '.png'))]
gr.Interface(
    inference, 
    [gr.inputs.Image(type="pil", label="Input")],
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch()
