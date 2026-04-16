import boto3
import time
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

CONFIG = {
    "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "temperature": 1.0,
    "max_tokens": 1000,
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

with open("claude_model_attributes.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

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
                system=[{"text": CONFIG["system_prompt"]}],
                inferenceConfig={
                    "temperature": CONFIG["temperature"],
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
            "max_tokens": CONFIG["max_tokens"],
            "system_prompt": CONFIG["system_prompt"]
        })

        with open("claude_resp_with_systemPrompt.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        time.sleep(2)

    messages = []

print(f"\nDone. {len(all_results)} results saved to claude_resp_with_systemPrompt.json")
