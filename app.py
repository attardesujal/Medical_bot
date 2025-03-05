import os
import time
import tkinter as tk
from tkinter import scrolledtext, ttk
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 10000}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

def invoke_with_retry(chain, query, max_retries=2, wait_time=1):
    for attempt in range(max_retries):
        try:
            response = chain.invoke({'query': query})
            return response
        except Exception as e:
            if '500' in str(e):
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")

# GUI Implementation
class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Medico - AI Medical Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#001f3f")

        # Style Configuration
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12), padding=5)
        self.style.configure("TEntry", font=("Arial", 14))
        self.style.configure("TFrame", background="#001f3f")

        # Main Frame
        self.main_frame = ttk.Frame(root, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, state=tk.DISABLED,
                                                      bg="#001f3f", fg="white", font=("Arial", 12))
        self.chat_display.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        # Input Frame
        self.input_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.input_frame.pack(pady=10, fill=tk.X, padx=20)

        # Input Box (Centered when empty)
        self.user_input = ttk.Entry(self.input_frame, font=("Arial", 14))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.user_input.bind("<Return>", self.send_message)

        # Send Button
        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message, width=10)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Clear Chat Button
        self.clear_button = ttk.Button(self.main_frame, text="Clear Chat", command=self.clear_chat, width=15)
        self.clear_button.pack(pady=5)

        # Set focus on input
        self.user_input.focus_set()

    def send_message(self, event=None):
        user_query = self.user_input.get().strip()
        if user_query:
            self.display_message("You", user_query, "#004080")  # Dark blue for user text
            self.user_input.delete(0, tk.END)  # Clear input field

            response = invoke_with_retry(qa_chain, user_query)
            bot_response = response["result"]
            self.display_message("Medico", bot_response, "#D3D3D3")  # Light gray for bot response

    def display_message(self, sender, message, bg_color):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}:\n", "bold")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)  # Auto-scroll to the latest message

    def clear_chat(self):
        """Clears all chat history."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)

# Run the application
root = tk.Tk()
chat_app = ChatApplication(root)
root.mainloop()
