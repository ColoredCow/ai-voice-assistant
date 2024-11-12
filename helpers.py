import os

def getModelDirectoryPath():
    current_path = os.path.abspath(__file__)
    project_root = current_path[:current_path.index("ai-voice-assistant") + len("ai-voice-assistant")]
    return project_root

