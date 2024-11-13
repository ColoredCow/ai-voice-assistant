import json
import ollama
import glob

# Directory containing training data files
training_data_path = 'fine-tuning/data/*1.json'

# Load training data from multiple JSON files
training_data = []
for filepath in glob.glob(training_data_path):
    with open(filepath, 'r') as file:
        data = json.load(file)
        training_data.extend(data)  # Assuming each file contains a list of Q&A pairs

# Format the training data as a JSON string
training_data_json = json.dumps(training_data, indent=2)

# Define the modelfile content with the loaded training data
modelfile_content = f"""
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM \"\"\"
You are an agriculutural assistant for farmers in India. Form the answer based on the following mentioned data otherwise give a generic answer:
{training_data_json}
\"\"\"
"""

print(modelfile_content)

ollama.create(model='paani', modelfile=modelfile_content)


# Initialize the chatbot
while True:
    # Prompt the user for input
    user_input = input("\nYou: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break

    # Get and print the chatbot response
    stream = ollama.chat(
        model='paani',
        messages=[{'role': 'user', 'content': user_input}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)