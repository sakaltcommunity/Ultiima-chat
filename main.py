import gradio as gr
from huggingface_hub import InferenceClient

# 使用可能なモデルのリスト
models = ["Sakalti/Saba1.5-Pro", "Sakalti/light-3b-beta", "Sakalti/Neptuno-Alpha", "Sakalti/light-3B", "Sakalti/Lunar-Bfloat16-4B", "Sakalti/tara-3.8B", "Sakalti/Lunar-4B", "Qwen/QwQ-32B-Preview", "Sakalti/SJT-4B-v1.1"]

# システムメッセージのテンプレート
system_message_templates = {
    "架空のキャラ　ナナ": "あなたの名前はナナ。優しい架空のキャラクターのaiとして。",
    "架空のキャラ　アオイ": "あなたの名前はアオイ。常識的な考えを持っている。語尾は「だぜ」",
}

def update_system_message(selected_template):
    return system_message_templates.get(selected_template, "あなたはフレンドリーなチャットボットです。")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    selected_model
):
    # 型変換: selected_modelを文字列に変換
    selected_model = str(selected_model)
    
    # 選択したモデルに基づいてInferenceClientを初期化
    client = InferenceClient(selected_model)

    messages = []

    # SJT-2.5Bの場合は、systemメッセージを最初のユーザーのメッセージに埋め込む
    if selected_model == "Sakalti/SJT-2.5B":
        if not history:
            message = f"{system_message}\n{message}"
        else:
            messages.append({"role": "user", "content": system_message})
    
    # 通常のsystemメッセージとして送信
    else:
        messages.append({"role": "system", "content": system_message})

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token

    return response

# インターフェース
with gr.Blocks() as demo:
    gr.Markdown("# チャットボット")
    
    system_message_template_dropdown = gr.Dropdown(
        choices=list(system_message_templates.keys()),
        value="架空のキャラ　ナナ",
        label="システムメッセージテンプレート"
    )
    
    system_message_textbox = gr.Textbox(
        value=system_message_templates["架空のキャラ　ナナ"],
        label="システムメッセージ"
    )
    
    system_message_template_dropdown.change(update_system_message, inputs=system_message_template_dropdown, outputs=system_message_textbox)
    
    chat_interface = gr.ChatInterface(
        respond,
        additional_inputs=[
            system_message_textbox,
            gr.Slider(minimum=1, maximum=2048, value=768, step=1, label="新規トークン最大"),
            gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="温度"),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p (核 sampling)",
            ),
            gr.Dropdown(choices=models, value=models[0], label="モデル"),
        ],
        concurrency_limit=30  # 例: 同時に30つのリクエストを処理
    )

if __name__ == "__main__":
    demo.launch(share=True)
