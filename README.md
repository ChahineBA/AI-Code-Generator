# Code Reader & Generator 🚀

This project allows you to read and analyze **PDF files** (like GitHub READMEs) and **code files** 📝. It can **generate code**, **fix bugs**, and **save the results** to new files 💻. 

## Features ✨
- 📖 Read and analyze PDF files (e.g., GitHub project READMEs)
- 🔧 Generate and fix code
- 💾 Save generated/fixed code to new files
- 🔍 Create vector stores using **Hugging Face embeddings**

## Technologies 💡
- **LlamaIndex** for indexing and querying 🔍
- **Mistral** and **CodeLlama** LLMs for code generation and fixing 🤖
- **Hugging Face embeddings** for vector store creation 🧠

## API Keys Required 🔑
To run this project, you will need to set up the following API keys in your `.env` file:
- **Mistral API Key**: For using the Mistral LLM
- **Hugging Face API Key**: For accessing Hugging Face embeddings and models
- **LlamaIndex Cloud API Key**: For LlamaIndex integrations

### Example .env File:
```env
MISTRAL_API_KEY=your_mistral_api_key
HUGGING_FACE_TOKEN=your_hugging_face_api_key
LLAMA_CLOUD_API_KEY=your_llamaindex_api_key
```
## Setup 🛠️
1. Clone the repository
2. Install dependencies with pip install -r requirements.txt
3. Set up your .env file with the necessary API keys
4. Run the script to start reading and generating code!
