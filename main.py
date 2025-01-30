import gradio as gr
import requests

# Vercel APIのエンドポイント
API_URL = "https://ultiima-api-vercel.app/api-sakal-models"  # デプロイしたAPIのURLを記入

# APIリクエスト関数
def generate_text(prompt, max_new_tokens, temperature, top_p, top_k):
    # APIにデータを送信
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
    }
    response = requests.post(API_URL, json=payload)
    
    # APIのレスポンスを処理
    if response.status_code == 200:
        return response.json().get("generated_text", "No text generated.")
    else:
        return f"Error: {response.status_code}, {response.text}"

# Gradioインターフェース
inputs = [
    gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=2),
    gr.Slider(label="Max New Tokens", minimum=10, maximum=200, step=10, value=50),
    gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=1.0),
    gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, step=0.1, value=1.0),
    gr.Slider(label="Top-k", minimum=10, maximum=100, step=10, value=50),
]

outputs = gr.Textbox(label="Generated Text")

app = gr.Interface(
    fn=generate_text,
    inputs=inputs,
    outputs=outputs,
    title="Text Generation App",
    description="Generate text using your deployed API with adjustable parameters.",
)

# アプリを起動
if __name__ == "__main__":
    app.launch()
