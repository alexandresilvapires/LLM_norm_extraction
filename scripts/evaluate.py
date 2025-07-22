import argparse
from pathlib import Path
import json
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm


main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))
from LLMs import init_llm
from prompt_dataset.create_dataloader import NormExtraction, collate_fn


def write_str_to_file(filename: str, text: str) -> None:
    """
    Writes the given text (str) to a file specified by 'filename'.
    If the file exists, it is overwritten.

    :param filename: The path to the file where the text should be saved.
    :param text: The string that will be written to the file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def read_text_file(file_path):
    """
    Read the contents of a text file and return them as a string.

    Args:
    file_path (str): The path to the text file.

    Returns:
    str: The contents of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"

def evaluate_model(model_name: str, temperature: float, max_new_tokens: int, output_folder: str, dataset_path: Path):
    output_folder = output_folder / Path(f"{model_name if '/' not in model_name else model_name.split('/')[-1]}_{dataset_path.stem}")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f'responses_{model_name if "/" not in model_name else model_name.split("/")[-1]}_{dataset_path.stem}.json'
    response_folder = output_folder / "responses"
    response_folder.mkdir(parents=True, exist_ok=True)

    dataset = NormExtraction(json_file_path=dataset_path)

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Adjust batch_size as needed
        collate_fn=collate_fn,
        shuffle=False
    )
    model = init_llm(model_name, temperature, max_new_tokens)
    results = []
    for idx, batch in enumerate(tqdm(dataloader)):
        response_file = Path(response_folder / f"{idx}_response.txt")
        prompt_dict = batch[0]  # we always use batch size of one
        if response_file.exists():
            response = read_text_file(response_file)
        else:
            response = model.predict(prompt_dict['prompt'])
            write_str_to_file(response_file, response)
            write_str_to_file(response_folder / f"{idx}_prompt.txt", prompt_dict['prompt'])
        prompt_dict['response'] = response
        prompt_dict['model_name'] = model_name
        results.append(prompt_dict)


    # Write the list of dicts to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)




def main():
    # 1. Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate a Large Language Model on a custom benchmark."
    )

    # 2. Add required arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-3.5-turbo",
        help="Name or path of the model to evaluate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for text generation. Lower values (e.g., 0.2) \
              make outputs more deterministic, higher values (e.g., 1.0) make outputs more random."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate in response to a prompt."
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default="outputs",
        help="Folder where evaluation results will be saved."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="prompt_dataset/prompt_dataset.json",
        help="Path to Dataset."
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Print out or log the arguments for confirmation
    print("[main] Received the following arguments:")
    print(f"  Model Name      : {args.model_name}")
    print(f"  Temperature     : {args.temperature}")
    print(f"  Max New Tokens  : {args.max_new_tokens}")
    print(f"  Output Folder   : {args.output_folder}")
    print(f"  Dataset   : {args.dataset}")

    print("")

    # 5. Call the separate evaluation function
    evaluate_model(
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_folder=args.output_folder,
        dataset_path=args.dataset
    )


if __name__ == "__main__":
    main()
