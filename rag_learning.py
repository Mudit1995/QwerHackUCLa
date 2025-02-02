# app.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import fitz
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from string import Template
import ollama
import os

app = Flask(__name__)
CORS(app)

# Initialize RAG system
class RAGSystem:
    def __init__(self, pdf_directory):
        self.documents = self.load_documents(pdf_directory)
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index = self.create_faiss_index()
        self.retriever = SimpleRetriever(self.index, self.embeddings_model, self.documents)

    def load_documents(self, directory):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                try:
                    text = self.extract_text_from_pdf(file_path)
                    documents.append(text)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        return documents

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text

    def create_faiss_index(self):
        document_embeddings = np.array(
            self.embeddings_model.embed_documents(self.documents)
        ).astype('float32')
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(document_embeddings)
        return index

class SimpleRetriever:
    def __init__(self, index, embeddings_model, documents):
        self.index = index
        self.embeddings_model = embeddings_model
        self.documents = documents

    def retrieve(self, query, k=3):
        query_embedding = self.embeddings_model.embed_query(query)
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return [self.documents[i] for i in indices[0]]

# Initialize RAG system
rag = RAGSystem('/Users/muditmohan/Desktop/RAGImplemnentaionofpdf')

prompt_template = Template("""
Use ONLY the context below.
If unsure, say "I don't know".
Keep answers under 4 sentences.

Context: $context
Question: $question
Answer:
""")

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LGBTQ+ Support Chat</title>
        <style>
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .chat-box {
                height: 500px;
                border: 1px solid #ccc;
                padding: 20px;
                margin-bottom: 20px;
                overflow-y: auto;
                background: #f9f9f9;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background: #e3f2fd;
                margin-left: 20%;
            }
            .bot-message {
                background: #f5f5f5;
                margin-right: 20%;
            }
            input[type="text"] {
                width: 70%;
                padding: 10px;
                margin-right: 10px;
            }
            button {
                padding: 10px 20px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>🌈 LGBTQ+ Support Chat</h1>
            <div class="chat-box" id="chatBox"></div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Ask about LGBTQ+ support...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function appendMessage(text, isUser) {
                const chatBox = document.getElementById('chatBox');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                messageDiv.textContent = text;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const question = input.value.trim();
                
                if (!question) return;
                
                appendMessage(question, true);
                input.value = '';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    appendMessage(data.answer, false);
                } catch (error) {
                    appendMessage('Error connecting to the server', false);
                }
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    
    # Retrieve context
    context = rag.retriever.retrieve(question)
    combined_context = "\n".join(context)
    
    # Generate response
    response = ollama.generate(
        model="deepseek-r1:1.5b",
        prompt=prompt_template.substitute(
            context=combined_context, 
            question=question
        )
    )
    
    return jsonify({
        'question': question,
        'answer': response["response"].strip()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
