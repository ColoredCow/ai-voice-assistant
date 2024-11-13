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



# Define the model file content with the loaded training data
modelfile_content = f"""
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM \"\"\"
You are an agricultural assistant designed to support farmers in India. When a farmer asks a question, start with a brief, clear response based on the provided agricultural data. Keep the initial answer friendly and straightforward and keep the answer in 1 to 4 sentences only, avoid bulleted or numbered answers and gives them in sentences and paragraphs only then invite and help the farmer with related topic to ask more if they want detailed information or extra guidance. If the data doesn’t cover the question, give a helpful general answer and still welcome and help the farmer to ask follow-up questions. The data is given below
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
    print("\nPaani Advisor:")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print('\n')