import boto3
import time
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

CONFIG = {
    "model": "us.meta.llama3-3-70b-instruct-v1:0",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1000,
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

        messages.append({"role": "user", "content": [{"text": prompt}]})

        try:
            response = client.converse(
                modelId=CONFIG["model"],
                messages=messages,
                inferenceConfig={
                    "temperature": CONFIG["temperature"],
                    "topP": CONFIG["top_p"],
                    "maxTokens": CONFIG["max_tokens"]
                }
            )
            answer = response["output"]["message"]["content"][0]["text"]
            finish_reason = response["stopReason"]
            messages.append({"role": "assistant", "content": [{"text": answer}]})
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
            "max_tokens": CONFIG["max_tokens"],
            "system_prompt": CONFIG["system_prompt"]
        })

        with open("llama_resp_no_systemPrompt.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        time.sleep(1)

    messages = []

print(f"\nDone. {len(all_results)} results saved to llama_resp_no_systemPrompt.json")
