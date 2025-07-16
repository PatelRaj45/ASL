import gradio as gr
import os
import torch
from model import create_resnet_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [asl_name.strip() for asl_name in f.readlines()]

# Fun facts / motivational messages for each class
fun_messages = {
    "A": "✊ A for Awesome!",
    "B": "🖐 B for Brilliant!",
    "C": "👌 C for Cool!",
    "D": "👉 D for Determined!",
    "E": "✋ E for Excellent!",
    "F": "🤞 F for Fantastic!",
    "G": "👈 G for Great!",
    "H": "🤚 H for Happy!",
    "I": "☝️ I for Incredible!",
    "J": "👋 J for Joyful!",
    "K": "🤟 K for Kind!",
    "L": "🦾 L for Lucky!",
    "M": "✌️ M for Magic!",
    "N": "🤙 N for Nice!",
    "O": "⭕ O for Outstanding!",
    "P": "🅿️ P for Powerful!",
    "Q": "🔍 Q for Quick!",
    "R": "®️ R for Rocking!",
    "S": "💪 S for Strong!",
    "T": "✝️ T for Talented!",
    "U": "⛎ U for Unique!",
    "V": "✌️ V for Victorious!",
    "W": "🤘 W for Wonderful!",
    "X": "❌ X for Xtraordinary!",
    "Y": "💥 Y for Youthful!",
    "Z": "⚡ Z for Zesty!",
    "del": "🚫 Delete!",
    "nothing": "😶 Nothing detected!",
    "space": "⬜ Space!"
}

# Create model
resnet_model, resnet_transforms = create_resnet_model(
    num_classes=len(class_names),
)

# Load weights
resnet_model.load_state_dict(
    torch.load(
        f="asl_resnet_model.pth",
        map_location=torch.device("cpu"),
    )
)

def predict(img) -> Tuple[Dict, float, str]:
    start_time = timer()
    img = resnet_transforms(img).unsqueeze(0)
    resnet_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(resnet_model(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    # Get top prediction and message
    top_class = class_names[torch.argmax(pred_probs)]
    message = fun_messages.get(top_class, "Great job!")
    return pred_labels_and_probs, pred_time, message

# Create examples grid with labels
example_list = []
for example in sorted(os.listdir("examples")):
    example_list.append([f"examples/{example}"])

title = "ASL Alphabet Vision ✋🤟"
description = """
Welcome to **ASL Alphabet Vision!  
Upload or click an example image to see the predicted ASL letter.  
The app predicts **A-Z, del, nothing, space** — and gives you a fun message! 🎉  
"""

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top Predictions"),
        gr.Number(label="Prediction Time (s)"),
        gr.Textbox(label="Fun Message")
    ],
    examples=example_list,
    title=title,
    description=description,
    allow_flagging="never"
)

demo.launch(share=True)
