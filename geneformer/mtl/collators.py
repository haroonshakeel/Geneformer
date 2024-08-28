# imports
import torch
from ..collator_for_classification import DataCollatorForGeneClassification
from . import TOKEN_DICTIONARY  # import the token dictionary from the mtl module's init

"""
Geneformer collator for multi-task cell classification.
"""

class DataCollatorForMultitaskCellClassification(DataCollatorForGeneClassification):
    class_type = "cell"

    def __init__(self, *args, **kwargs) -> None:
        # Use the loaded token dictionary from the mtl module's init
        super().__init__(token_dictionary=TOKEN_DICTIONARY, *args, **kwargs)

    def _prepare_batch(self, features):
        # Process inputs as usual
        batch = self.tokenizer.pad(
            features,
            class_type=self.class_type,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Check if labels are present
        if "label" in features[0]:
            # Initialize labels dictionary for all tasks
            labels = {task: [] for task in features[0]["label"].keys()}

            # Populate labels for each task
            for feature in features:
                for task, label in feature["label"].items():
                    labels[task].append(label)

            # Convert label lists to tensors, handling dictionaries appropriately
            for task in labels:
                if isinstance(labels[task][0], (list, torch.Tensor)):
                    dtype = torch.long
                    labels[task] = torch.tensor(labels[task], dtype=dtype)
                elif isinstance(labels[task][0], dict):
                    # Handle dict specifically if needed
                    pass  # Resolve nested data structure

            # Update the batch to include task-specific labels
            batch["labels"] = labels
        else:
            # If no labels are present, create empty labels for all tasks
            batch["labels"] = {
                task: torch.tensor([], dtype=torch.long)
                for task in features[0]["input_ids"].keys()
            }

        return batch

    def __call__(self, features):
        batch = self._prepare_batch(features)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.clone().detach()
            elif isinstance(v, dict):
                # Assuming nested structure needs conversion
                batch[k] = {
                    task: torch.tensor(labels, dtype=torch.int64)
                    for task, labels in v.items()
                }
            else:
                batch[k] = torch.tensor(v, dtype=torch.int64)

        return batch
