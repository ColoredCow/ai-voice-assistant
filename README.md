# AI Voice Assistant

### Prerequisites
- Python 3.10 or 3.11 (not compatible with Python 3.13)

### Installation
1. Set up a Python virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate
    ```
2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Acquire a Hugging Face API token:
   - Sign up at [huggingface.co](https://huggingface.co/) if you don't already have an account.
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Generate a new token (or use an existing one) and copy it.
4. Set up your `.env` file
    ```sh
    cp .env.example .env
    ```
5. Set the hugging face token in the env file.
6. Run the Flask application:
    ```sh
    flask --debug run
    ```

7. Open the web server in your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)
