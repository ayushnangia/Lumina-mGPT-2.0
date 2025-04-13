import os
import json
from pathlib import Path
import argparse

try:
    from datasets import load_dataset
except ImportError:
    print("The 'datasets' library is required. Please install it:")
    print("pip install datasets")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("The 'Pillow' library is required. Please install it:")
    print("pip install Pillow")
    exit(1)

# --- Configuration ---
DATASET_NAME = "julianmoraes/doodles-captions-manual"
DATASET_SPLIT = "train"  # Assuming you want the 'train' split
IMAGE_COLUMN = "image"    # Assumed column name for image data
TEXT_COLUMN = "text"      # Assumed column name for caption data

# Get the root directory of the project (assuming this script is in Lumina-mGPT-2.0/scripts)
PROJECT_ROOT = Path(__file__).parent.parent

# --- Default Paths (relative to project root) ---
DEFAULT_IMAGE_SAVE_DIR = PROJECT_ROOT / "datasets" / "doodles"
DEFAULT_OUTPUT_JSON_PATH = PROJECT_ROOT / "pre_tokenize" / "doodles_input.json"
# --- End Default Paths ---

def prepare_dataset(image_save_dir: Path, output_json_path: Path):
    """Loads the dataset, saves images, and creates the JSON input file."""
    print(f"Loading dataset '{DATASET_NAME}'...")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        print(f"Dataset loaded successfully. Found {len(dataset)} examples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the dataset name and ensure you have network connectivity.")
        exit(1)

    # Ensure the image save directory exists
    print(f"Ensuring image save directory exists: {image_save_dir}")
    image_save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the output directory exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = []
    total_examples = len(dataset)

    print("Processing dataset examples...")
    for i, example in enumerate(dataset):
        try:
            image_obj = example[IMAGE_COLUMN]
            caption = example[TEXT_COLUMN]

            if not isinstance(image_obj, Image.Image):
                print(f"Warning: Expected PIL Image in column '{IMAGE_COLUMN}' for example {i}, but got {type(image_obj)}. Skipping.")
                continue

            if not isinstance(caption, str):
                print(f"Warning: Expected string in column '{TEXT_COLUMN}' for example {i}, but got {type(caption)}. Skipping.")
                continue

            # Get image dimensions
            width, height = image_obj.size

            # Define image filename (using index with padding)
            image_filename = f"{i:06d}.png" # e.g., 000000.png, 000001.png
            image_save_path = image_save_dir / image_filename

            # Save the image
            image_obj.save(image_save_path, format="PNG")

            # Format the conversation entry
            conversation_entry = {
                "conversations": [
                    {
                        "from": "human",
                        # Including {h}x{w} as per TRAIN.md example format
                        "value": f"Generate an image of {height}x{width} according to the following prompt:<|prompt|>{caption}"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    },
                ],
                # Store the absolute path to the saved image
                "image_path": str(image_save_path.resolve()),
                # Add the prompt as a top-level key for the pre-tokenizer
                "prompt": caption
            }
            output_data.append(conversation_entry)

            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == total_examples:
                print(f"Processed {i + 1}/{total_examples} examples...")

        except KeyError as e:
            print(f"Error accessing column '{e}' in example {i}. Please verify the assumed column names ('{IMAGE_COLUMN}', '{TEXT_COLUMN}'). Skipping example.")
            continue
        except Exception as e:
            print(f"Error processing example {i}: {e}. Skipping.")
            continue

    # Save the output JSON file
    print(f"Saving formatted data to {output_json_path}...")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print("Successfully created JSON input file.")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Prepare Hugging Face dataset for Lumina pre-tokenization.")
    parser.add_argument(
        "--image-save-dir",
        type=Path,
        default=DEFAULT_IMAGE_SAVE_DIR,
        help=f"Directory to save downloaded images. Defaults to: {DEFAULT_IMAGE_SAVE_DIR}"
    )
    parser.add_argument(
        "--output-json-path",
        type=Path,
        default=DEFAULT_OUTPUT_JSON_PATH,
        help=f"Path to save the output JSON file. Defaults to: {DEFAULT_OUTPUT_JSON_PATH}"
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    prepare_dataset(args.image_save_dir, args.output_json_path) 