from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_parse import LlamaParse
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.agent import ReActAgent
import os
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
import ast

# Load environment variables from a .env file
load_dotenv()

# Get the Hugging Face token from environment variables
HF_token = os.getenv("HUGGING_FACE_TOKEN")

# Initialize the LLM using Hugging Face Inference API with the specified model and token
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.3", token=HF_token)

# Define a parser for extracting content from documents, outputting results in Markdown format
parser = LlamaParse(result_type="markdown")

# Map file types to specific parsers for content extraction
file_extractor = {".pdf": parser}

# Load documents from the './data' directory using the specified file extractors
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Resolve and initialize the embedding model for vectorization
embed_model = resolve_embed_model(embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"))

# Create a vector index from the loaded documents using the embedding model
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Set up a query engine to process user queries using the vector index and LLM
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools for the agent, including a query engine for API documentation and a code reader
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This provides documentation about an API. Use this for reading API docs."
        )
    ),
    code_reader  # Assume this tool reads and analyzes code
]

# Initialize another LLM for processing code, with a lower temperature for deterministic outputs
code_llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)

# Create a reactive agent that uses the defined tools and context
agent = ReActAgent.from_tools(tools=tools, llm=code_llm, verbose=True, context=context)

# Define a Pydantic model for the structure of the expected output
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

# Set up a parser for generating structured output using the Pydantic model
parser = PydanticOutputParser(CodeOutput)

# Format the output using a custom template
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)

# Create a query pipeline for chaining prompts and LLMs
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Main loop to interact with the user
while (prompt := input('Enter a prompt: ')) != "q":
    retries = 0
    while retries < 3:
        try:
            # Query the agent with the user's prompt
            result = agent.query(prompt)
            
            # Process the result using the query pipeline
            next_result = output_pipeline.run(response=result)
            
            # Clean and parse the result as a JSON-like dictionary
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
        except Exception as e:
            # Retry logic in case of errors
            retries += 1
            print(f'Error occurred, retry #{retries}:', e)
    
    if retries >= 3:
        # Notify the user if retries exceed the limit
        print("Unable to process request, try again...")
        continue

    # Display and handle the generated code and its metadata
    print("Code Generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json['description'])
    filename = cleaned_json['filename']
    
    try:
        # Save the generated code to a file
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json['code'])
            print('Saved File', filename)
    except:
        # Handle errors during file saving
        print("Error Saving File...")
