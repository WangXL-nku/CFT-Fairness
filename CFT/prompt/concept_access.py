# main.py
# Main entry point for CFT bias audit. Integrates prompt templates and model clients.

import os
import argparse
import base64
from openai import OpenAI

from prompts import get_prompt, TASK_DESCRIPTIONS, DEFAULT_TASK
from models import create_client

# Mapping from legacy model_id (0-3) to task names.
MODEL_ID_TO_TASK = {
    0: "Big Nose",       # not in paper; adjust as needed
    1: "Heavy Makeup",
    2: "Gender",
    3: "Gender",
}

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_concept_img_path(concept_id, img_base_dir, model_name, img_num=20):
    """
    Construct image file paths for a given concept.
    Follows the exact folder structure of the original code.
    """
    concept_path = os.path.join(img_base_dir, model_name, f"Concept{concept_id}")
    concept_crop_path_list = [
        os.path.join(concept_path, f"{model_name}_crop_{concept_id}_{j}.png")
        for j in range(img_num)
    ]
    concept_image_path_list = [
        os.path.join(concept_path, f"{model_name}_img_{concept_id}_{j}.png")
        for j in range(img_num)
    ]
    concept_image_crop_path_list = [
        os.path.join(concept_path, f"{model_name}_img_crop_{concept_id}_{j}.png")
        for j in range(img_num)
    ]
    return concept_crop_path_list, concept_image_path_list, concept_image_crop_path_list

def build_messages_for_concept(image_paths, task_name):
    """
    Build the messages payload for the MLLM API.
    """
    prompt_text = get_prompt(task_name)
    content = [
        {"type": "text", "text": prompt_text}
    ]
    for path in image_paths:
        if os.path.exists(path):
            b64 = encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
            })
        else:
            print(f"Warning: image not found - {path}")
    return [{"role": "user", "content": content}]

def main():
    parser = argparse.ArgumentParser(description="CFT Bias Audit – Unified Entry")
    parser.add_argument("-img_base_dir", type=str, default="/concept_images",
                        help="Base directory containing concept images.")
    parser.add_argument("-model_id", type=int, default=None,
                        help="Legacy model index (0-3). If provided, task is derived from this.")
    parser.add_argument("-task", type=str, default=None,
                        help="Explicit task name (e.g., 'Blonde Hair'). Overrides model_id mapping.")
    parser.add_argument("-model_type", type=str, required=True, choices=["qwen", "glm"],
                        help="Which multimodal model to use: qwen or glm.")
    parser.add_argument("-img_num", type=int, default=20,
                        help="Number of images per concept.")
    parser.add_argument("-concept_ids", type=int, nargs="+", default=[1,16,17,18],
                        help="List of concept IDs to audit.")
    parser.add_argument("-gpu", type=int, default=0,
                        help="GPU device (unused, kept for compatibility).")
    args = parser.parse_args()

    # Determine the task name
    if args.task:
        task_name = args.task
    elif args.model_id is not None:
        model_display_name = [
            'Big_Nose-1', 'Heavy_Makeup-1', 'Male-1', 'Male-0'
        ][args.model_id]
        task_name = MODEL_ID_TO_TASK.get(args.model_id, DEFAULT_TASK)
    else:
        task_name = DEFAULT_TASK

    print(f"Auditing for task: {task_name}")
    print(f"Using model: {args.model_type}")

    # Create model client
    client, model_name = create_client(args.model_type)

    # Determine the folder name (legacy model display name)
    # For consistency with original folder structure, we still need the 4-element list.
    if args.model_id is not None:
        model_display_name = [
            'Big_Nose-1', 'Heavy_Makeup-1', 'Male-1', 'Male-0'
        ][args.model_id]
    else:
        # If no model_id, we assume a default folder name (can be overridden by user)
        model_display_name = f"{task_name.replace(' ', '_')}-1"
        print(f"Using folder name: {model_display_name} (may need adjustment)")

    # Process each concept
    for concept_id in args.concept_ids:
        print(f"\n—————————— Processing Concept {concept_id} ——————————")
        _, _, concept_image_crop_path_list = get_concept_img_path(
            concept_id, args.img_base_dir, model_display_name, args.img_num
        )
        existing_images = [p for p in concept_image_crop_path_list if os.path.exists(p)]
        if not existing_images:
            print(f"No images found for Concept {concept_id}, skipping.")
            continue

        messages = build_messages_for_concept(existing_images, task_name)

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            result = completion.choices[0].message.content
            print(result)
        except Exception as e:
            print(f"API call failed for Concept {concept_id}: {e}")

if __name__ == "__main__":
    main()