import google.generativeai as genai
import time
import json

genai.configure(api_key="AQs_0")

CONFIG = {
    "model": "gemini-2.5-flash",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_output_tokens": 1000,
    "seed": 42,
    "system_prompt": "Respond naturally. If unclear, ask for clarification."
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

with open("gemini_model_attributes.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

all_results = []

for cat_name, prompts in categories.items():
    print(f"\n--- Running category: {cat_name} ({len(prompts)} prompts) ---")

    model = genai.GenerativeModel(
        model_name=CONFIG["model"],
        system_instruction=CONFIG["system_prompt"],
        generation_config=genai.GenerationConfig(
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            top_k=CONFIG["top_k"],
            presence_penalty=CONFIG["presence_penalty"],
            frequency_penalty=CONFIG["frequency_penalty"],
            max_output_tokens=CONFIG["max_output_tokens"]
        )
    )
    chat = model.start_chat(history=[])

    for i, prompt in enumerate(prompts):
        print(f"  {cat_name} prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

        try:
            response = chat.send_message(prompt)
            answer = response.text
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "UNKNOWN"
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
            "top_k": CONFIG["top_k"],
            "presence_penalty": CONFIG["presence_penalty"],
            "frequency_penalty": CONFIG["frequency_penalty"],
            "max_output_tokens": CONFIG["max_output_tokens"],
            "system_prompt": CONFIG["system_prompt"]
        })
        time.sleep(1)

with open("gemini_resp_with_systemPrompt.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(all_results)} results saved to gemini_resp_with_systemPrompt.json")
