from gradio_client import Client
import gradio as gr
from time import sleep

client1 = Client("http://127.0.0.1:7860/")
client2 = Client("http://127.0.0.1:7864/")

def gen_client(model_choice1, model_choice2, prompt, temperature):
    # 使用 client1 生成结果
    result1 = client1.predict(
        model_choice=model_choice1,
        prompt=prompt,
        temperature=temperature,
        api_name="/predict"
    )

    # 使用 client2 生成结果
    result2 = client2.predict(
        model_choice=model_choice2,
        prompt=prompt,
        temperature=temperature,
        api_name="/predict"
    )

    # return [result1, result2]

    # 将两个结果组合成逐字符输出
    display_out1 = ""
    display_out2 = ""
    max_len = max(len(result1), len(result2))

    for i in range(max_len):
        if i < len(result1):
            display_out1 += result1[i]
        if i < len(result2):
            display_out2 += result2[i]
        
        yield [display_out1, display_out2]
        sleep(0.05)

model_choices1 = gr.Dropdown(choices=["Pretrain", "SFT"], label="选择模型 (Client 1)")
model_choices2 = gr.Dropdown(choices=["Pretrain", "SFT"], label="选择模型 (Client 2)")
temperature_slider = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label="Temperature")
inputs = [model_choices1, model_choices2, gr.Textbox(lines=5, placeholder="输入prompt"), temperature_slider]
outputs = [gr.Textbox(label="Client1"), gr.Textbox(label="Client2")]
interface = gr.Interface(fn=gen_client, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    interface.launch(share=True)
