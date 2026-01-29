import gradio as gr  #create web interface 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch   #requied to backend

# Load fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(user_input):
    input_text = f"Customer: {user_input}\nSupport:"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Support:" in decoded:
        return decoded.split("Support:")[-1].strip()
    else:
        return decoded

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Customer Question"),
    outputs=gr.Textbox(label="Support Response"),
    title="Customer Support Chatbot (Fine-Tuned GPT-2)"
)

if __name__ == "__main__":
    interface.launch()
