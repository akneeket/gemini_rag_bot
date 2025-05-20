
# 🤖 Gemini RAG Bot

**Gemini RAG Bot** is an AI-powered chatbot that allows users to ask questions about the content of uploaded **PDF**, **CSV**, **JSON**, or even **webpage URLs**. It leverages **Google's Gemini Pro** and **FAISS-based vector retrieval** to deliver accurate, context-aware answers from documents.

![Gemini RAG Bot Banner](https://img.shields.io/badge/Built%20with-Gemini%20%7C%20LangChain%20%7C%20Streamlit-blueviolet)

---

## 🚀 Features

- 📄 Upload **PDF**, **CSV**, or **JSON** files
- 🌐 Enter a **website URL** for web-based RAG
- 🧠 Uses **FAISS** for fast document similarity search
- 🗣️ Natural language Q&A powered by **Gemini 1.5 Flash**
- 🔍 Intelligent document chunking and embedding with **HuggingFace Transformers**
- 💾 Persistent vectorstore for faster future queries

---

## 📸 Demo

![image](https://github.com/user-attachments/assets/fd2fbec1-220e-4901-a9fe-e50a657dfb62)


---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) - Frontend UI
- [LangChain](https://www.langchain.com/) - RAG framework
- [Google Gemini API](https://ai.google.dev/) - LLM-powered Q&A
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embedding model

---

## 📂 File Structure

```bash
gemini_rag_bot/
│
├── main.py                # Main Streamlit app
├── requirements.txt       # Dependencies
├── faiss_vectorstore.pkl  # Serialized vector store (optional)
├── README.md              # Project README
````

---

🛠️ Setup Instructions

    Clone the repo

git clone https://github.com/akneeket/gemini_rag_bot.git
cd gemini_rag_bot

    Create a virtual environment and activate it

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

    Install dependencies

pip install -r requirements.txt

    Add your Gemini API key

Edit the main.py file and replace:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

with your actual API key.

    Run the app

streamlit run main.py

❓ FAQs

    Can I use other file formats?

        Yes! CSV and JSON support is being added — stay tuned or contribute!

    Why do I get a FAISS pickle error?

        Delete the existing faiss_vectorstore.pkl if it's corrupted, and re-upload a document to regenerate it.

    Can I deploy this?

        Yes. You can deploy it using Streamlit Community Cloud, Render, or Docker.

🙌 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements and new features.
📃 License

This project is licensed under the MIT License. See the LICENSE file for details.

    Made with ❤️ by @akneeket


---

### ✅ Next Steps
- Save this as `README.md` in your project folder.
- Add & push it:

```bash
git add README.md
git commit -m "Add attractive README"
git push

