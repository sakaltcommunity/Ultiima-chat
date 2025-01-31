import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Sakalti/ultiima-78B"

# モデルとトークナイザの読み込み
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate(prompt, history, max_tokens, top_p, temperature, top_k, repetition_penalty):
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
    
    # テキスト生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 追加の入力パラメータを定義
additional_inputs = [
    gr.Slider(1, 2048, value=512, label="トークン最大長"),
    gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top P"),
    gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="温度"),
    gr.Slider(0, 100, value=50, label="Top K"),
    gr.Slider(1.0, 2.0, value=1.2, step=0.01, label="リペートペナルティ")
]

# インターフェースの作成と起動
chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=additional_inputs
)
chat_interface.launch(share=True)
