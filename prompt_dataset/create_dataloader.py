import json
import torch
from torch.utils.data import Dataset, DataLoader


class NormExtraction(Dataset):
    """
    A custom Dataset to load your benchmark data from a JSON file.
    Each entry is returned as a dictionary.
    """

    def __init__(self, json_file_path: str):
        """
        Args:
            json_file_path (str): Path to the JSON file containing your dataset.
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Return the sample at index `idx`. Each sample will be a dict with the
        following keys (based on your example):
          - name_donor
          - name_recipient
          - prior_reputation_recipient
          - action
          - topic
          - tag
          - prompt
        """
        sample = self.data[idx]

        return {
            "name_donor": sample["name_donor"],
            "name_recipient": sample["name_recipient"],
            "prior_reputation_recipient": sample["prior_reputation_recipient"],
            "action": sample["action"],
            "topic": sample["topic"],
            "donor_gender": sample["donor_gender"],
            "donor_region": sample["donor_region"],
            "recipient_gender": sample["recipient_gender"],
            "recipient_region": sample["recipient_region"],
            "tag": sample["tag"],
            "prompt": sample["prompt"]
        }

def collate_fn(samples):
    """
    A simple collate function that returns a list of dictionaries as is.
    If you'd like to batch process or tokenize, you can do it here.
    """
    return samples


# Example usage
if __name__ == "__main__":
    # Path to the JSON file containing your benchmark data
    data_file = "prompt_dataset.json"

    # Instantiate the dataset
    dataset = NormExtraction(json_file_path=data_file)

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Adjust batch_size as needed
        shuffle=True,  # Shuffle the data each epoch
        collate_fn=collate_fn
    )

    # 3. Example loop to go through the DataLoader
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch index: {batch_idx}")

        # Each 'batch' is a list of dictionaries (with length = batch_size)
        for sample_idx, sample_dict in enumerate(batch):
            print(f"  Sample index in batch: {sample_idx}")

            # Now you have direct access to the keys and values
            print("    name_donor:", sample_dict["name_donor"])
            print("    name_recipient:", sample_dict["name_recipient"])
            print("    prior_reputation_recipient:", sample_dict["prior_reputation_recipient"])
            print("    action:", sample_dict["action"])
            print("    topic:", sample_dict["topic"])
            print("    tag:", sample_dict["tag"])
            print("    prompt:", sample_dict["prompt"][:60], "...")  # Print first 60 chars for brevity

        # Optional: break early for demonstration
        if batch_idx == 1:
            break
