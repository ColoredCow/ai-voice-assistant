import json
import ollama
import glob

def get_training_data():
    # Directory containing training data files
    training_data_path = 'fine-tuning/data/*.text'

    # Initialize an empty string to hold all file contents
    training_data_str = ""

    # Read the content of each file and add it to the combined_data string
    for filepath in glob.glob(training_data_path):
        with open(filepath, 'r', encoding='utf-8') as file:
            training_data_str += file.read() + "\n"
    
    return training_data_str

def get_model_file():
    model_file_content = f"""
    FROM llama3.2

    # set the temperature to 1 [higher is more creative, lower is more coherent]
    PARAMETER temperature 1

    # set the system message
    SYSTEM \"\"\"
    You are an agricultural assistant designed to support farmers in India. When a farmer asks a question, start with a brief, clear response based on the provided agricultural data. Keep the answer friendly avoid bulleted points. Also always invite and suggest the farmer to ask more questions to related topic. If the data doesn’t cover the question, give a helpful general answer and still welcome and help the farmer to ask follow-up questions. The data is given below
    {get_training_data()}
    \"\"\"
    """
    return model_file_content

def setup_model(model_name):
    model_file_content = get_model_file()

    ollama.create(model=model_name, modelfile=model_file_content)
    print("Model is ready")

