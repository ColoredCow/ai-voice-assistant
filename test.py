from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
# model, tokenizer = load("/Users/coloredcow/.cache/lm-studio/models/mlx-community/Llama-3.2-3B-Instruct-4bit")

prompt="hello there how are you?"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print('response......')
print(response)
