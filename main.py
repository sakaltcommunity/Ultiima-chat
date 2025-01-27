import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Sakalti/SakalFusion-7B-Alpha"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def generate(prompt, history):
    messages = [
        {"role": "system", "content": "あなたはフレンドリーなチャットボットです。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=864
        temperature=0.7
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



chat_interface = gr.ChatInterface(
    fn=generate,
)
chat_interface.launch(share=True)
