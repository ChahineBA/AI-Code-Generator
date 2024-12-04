from llama_index.core.tools import FunctionTool  # Import the FunctionTool class to define a custom tool
import os  # Import the os module for working with file paths

# Define a function to read the contents of a code file given its filename
def code_reader_function(file_name):
    path = os.path.join('data', file_name)  # Construct the file path by joining the 'data' directory and filename
    try:
        # Attempt to open the file and read its content
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}  # Return the file content in a dictionary
    except Exception as e:
        # Handle any exceptions that occur and return an error message
        return {"error": str(e)}

# Define the `code_reader` tool using the FunctionTool class
code_reader = FunctionTool.from_defaults(
    fn=code_reader_function,  # Specify the function to be used by the tool
    name="code_reader",  # Assign a name to the tool
    description="""this tool can read the contents of code files and return 
    their results. Use this when you need to read the contents of a file""",  # Provide a description for the tool
)
