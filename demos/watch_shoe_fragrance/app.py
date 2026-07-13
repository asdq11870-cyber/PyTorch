import gradio as gr #pyright: ignore[reportMissingImports]
import torch #pyright: ignore[reportMissingImports]
import os
from model import create_effnetb2_model
from typing import Tuple, Dict
from timeit import default_timer as timer

class_names = ["watch", "shoe", "fragrance"]

effnetb2, effnetb2_transform = create_effnetb2_model(num_classes=len(class_names))
effnetb2.load_state_dict(torch.load(f="effnetb2.pth",map_location=torch.device("cpu")))

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    start_time = timer()

    img = effnetb2_transform(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time

example_list = [["examples/" + example] for example in os.listdir("examples")]
title = "Watch Shoe Fragrance Determinator!"
description = "Using an EfficientNetB2 feature extraction model"
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3, label="Predictions"), gr.Number(label="Prediction Time (s)")],
    title=title, description=description,examples=example_list
)
demo.launch()
