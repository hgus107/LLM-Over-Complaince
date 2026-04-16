import openai
import time
import json

client = openai.OpenAI(api_key="skz7MA")

CONFIG = {
    "model": "gpt-4.1-mini",
    "temperature": 1.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": 1000,
    "seed": 42,
    "system_prompt": "NONE"
}

categories = {
    "UNDERSPEC": [],
    "AMBIGUOUS": [],
    "CONTRADICTION": [],
    "NONSENSE": []
}

with open("Dataset_and_Instructions.txt", "r") as f:
    lines = f.readlines()

current_category = None
for line in lines:
    line = line.strip()
    if line.startswith("Category: ["):
        current_category = line.replace("Category: [", "").replace("]", "")
    elif current_category and line and line[0].isdigit() and ":" in line:
        prompt_text = line.split(":", 1)[1].strip()
        if prompt_text:
            categories[current_category].append(prompt_text)

all_results = []

for cat_name, prompts in categories.items():
    print(f"\n--- Running category: {cat_name} ({len(prompts)} prompts) ---")
    messages = []

    for i, prompt in enumerate(prompts):
        print(f"  {cat_name} prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=CONFIG["model"],
                messages=messages,
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                presence_penalty=CONFIG["presence_penalty"],
                frequency_penalty=CONFIG["frequency_penalty"],
                max_tokens=CONFIG["max_tokens"],
                seed=CONFIG["seed"]
            )
            answer = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            finish_reason = "ERROR"

        all_results.append({
            "category": cat_name,
            "prompt_number": i + 1,
            "prompt": prompt,
            "response": answer,
            "response_char_length": len(answer),
            "response_word_count": len(answer.split()),
            "finish_reason": finish_reason,
            "model": CONFIG["model"],
            "temperature": CONFIG["temperature"],
            "top_p": CONFIG["top_p"],
            "presence_penalty": CONFIG["presence_penalty"],
            "frequency_penalty": CONFIG["frequency_penalty"],
            "max_tokens": CONFIG["max_tokens"],
            "seed": CONFIG["seed"],
            "system_prompt": CONFIG["system_prompt"]
        })
        time.sleep(1)

    messages = []

with open("chatgpt_resp_no_systemPrompt.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(all_results)} results saved to chatgpt_resp_no_systemPrompt.json")
