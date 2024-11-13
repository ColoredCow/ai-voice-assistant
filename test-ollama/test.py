import json
import ollama
import glob

# Directory containing training data files
training_data_path = 'fine-tuning/data/*.text'

# Initialize an empty string to hold all file contents
training_data_str = ""

# Read the content of each file and add it to the combined_data string
for filepath in glob.glob(training_data_path):
    with open(filepath, 'r', encoding='utf-8') as file:
        training_data_str += file.read() + "\n"  # Append file content and a newline for separation

# Print the combined string with all JSON contents
print(training_data_str)

# Define the model file content with the loaded training data
modelfile_content = f"""
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM \"\"\"
You are an agricultural assistant designed to help farmers in India. When a farmer asks you a question, please formulate your response based on the provided agricultural data in a friendly way not formal way If the information is unavailable in the data, provide a helpful and generic answer. Always try to make conversation and suggest if they want to ask for more The data is added below
{training_data_str}
\"\"\"
"""

print("creating model")
ollama.create(model='paani', modelfile=modelfile_content)
print("Model Created")


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