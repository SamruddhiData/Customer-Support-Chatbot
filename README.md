# Customer-Support-Chatbot
Fine-tuned GPT-2 based Customer Support Chatbot using Python and Gradio
# Customer Support Chatbot (Fine-Tuned GPT-2)

This project is an end-to-end Generative AI application that demonstrates a Customer Support Chatbot built by fine-tuning the GPT-2 language model on custom customer support data.

The chatbot can answer common customer queries such as order tracking, returns, cancellations, refunds, and shipping-related questions.

---

## 🚀 Features
- Fine-tuned GPT-2 model on custom customer support Q&A dataset
- Handles real-world customer support queries
- Simple and interactive web interface using Gradio
- End-to-end pipeline: Training → Inference → Deployment

---

## 🛠 Tech Stack
- Python  
- Hugging Face Transformers  
- GPT-2  
- PyTorch  
- Gradio  

---

## 📂 Project Structure
Customer-Support-Chatbot/
│
├── data/
│ └── support_data.json
├── train.py
├── app.py
├── requirements.txt
├── screenshots/
│ └── chatbot_ui.png
└── README.md

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the Model
python train.py
3️⃣ Run the Chatbot
python app.py

