import datetime
from geneformer.classifier import Classifier
import geneformer.finetuner_utils

import numpy as np
import torch
import os
import random
import glob
import json



seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


import os

class FineTuner:
    def __init__(self, base_dir=None, aggregation_level=None, model_variant=None, 
                 task=None, dataset=None, model_version="V1"):
        """
        Initialize the FineTuner class with configuration parameters.
        
        Parameters:
        -----------
        base_dir : str, optional
            Base directory path. Defaults to "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer"
        aggregation_level : str, optional
            Aggregation level. Defaults to "metacell_2"
        model_variant : str, optional
            Model variant ("30M" or "95M"). Defaults to "30M"
        task : str, optional
            Task type. "dosage_sensitivity" or "disease_classification". Defaults to "disease_classification"
        dataset : str, optional
            Dataset name. One of these for "disease_classification": ["genecorpus_heart_disease", "cellnexus_blood_disease", "cellnexus_covid_disease"]. Defaults to "cellnexus_covid_disease"
            or use "genecorpus_dosage_sensitivity" for "dosage_sensitivity" task.
        model_version : str, optional
            Model version. Defaults to "V1". Currently not used but can be extended for future use.
        num_crossval_splits : int, optional
            Number of cross-validation splits. Can be 1 or 5. Defaults to 1.
        """
        
        # Set default values
        self.BASE_DIR = base_dir or "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer"
        self.AGGREGATION_LEVEL = aggregation_level or "metacell_2"
        self.MODEL_VARIANT = model_variant or "30M"
        self.TASK = task or "disease_classification"
        self.DATASET = dataset or "cellnexus_covid_disease"
        self.MODEL_VERSION = model_version

        self.DATASET_PATH = os.path.join(self.BASE_DIR, "datasets", self.TASK, self.DATASET)

        print (f"Cuda available: {torch.cuda.is_available()}")
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 0:
            print (f"Using {self.num_gpus} GPU(s)")
        else:
            print ("No GPUs available, using CPU")

        
        # Valid combinations
        self.VALID_COMBINATIONS = {
            "disease_classification": ["genecorpus_heart_disease", "cellnexus_blood_disease", "cellnexus_covid_disease"],
            "dosage_sensitivity": ["genecorpus_dosage_sensitivity"]
        }
        
        # Validate inputs
        self._validate_inputs()
        
        # Set file paths based on model variant
        self._set_file_paths()
        
        # Set classifier type based on task
        self._set_classifier_type()
        
        # Create output directory
        self._create_output_directory()
        
        # Print verification
        self._print_verification()
    
    def _validate_inputs(self):
        """Validate task and dataset combinations."""
        assert self.TASK in self.VALID_COMBINATIONS, f"Unknown TASK: '{self.TASK}'"
        assert self.DATASET in self.VALID_COMBINATIONS[self.TASK], \
            f"For TASK='{self.TASK}', DATASET must be one of {self.VALID_COMBINATIONS[self.TASK]}, got '{self.DATASET}'"
    
    def _set_file_paths(self):
        """Set file paths based on model variant."""
        if self.MODEL_VARIANT == "30M":
            self.GENE_MEDIAN_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
            )
            self.TOKEN_DICTIONARY_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
            )
            self.ENSEMBL_MAPPING_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
            )
        elif self.MODEL_VARIANT == "95M":
            self.GENE_MEDIAN_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/gene_median_dictionary_gc95M.pkl"
            )
            self.TOKEN_DICTIONARY_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/token_dictionary_gc95M.pkl"
            )
            self.ENSEMBL_MAPPING_FILE = os.path.join(
                self.BASE_DIR, "Geneformer/geneformer/ensembl_mapping_dict_gc95M.pkl"
            )
        else:
            raise ValueError("MODEL_VARIANT must be either '30M' or '95M'")
    
    def _set_classifier_type(self):
        """Set classifier type based on task."""
        if self.TASK == "disease_classification":
            self.classifier_type = "cell"
        elif self.TASK == "dosage_sensitivity":
            self.classifier_type = "gene"
        else:
            raise ValueError("TASK must be either 'disease_classification' or 'dosage_sensitivity'")
    
    def _create_output_directory(self):
        """Create output directory."""
        self.output_dir = os.path.join(
            self.BASE_DIR, 
            "trained_cell_classification_models", 
            self.TASK, 
            self.DATASET, 
            f"{self.MODEL_VARIANT}_{self.AGGREGATION_LEVEL}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _print_verification(self):
        """Print file paths for verification."""
        print(f"GENE_MEDIAN_FILE: {self.GENE_MEDIAN_FILE}")
        print(f"TOKEN_DICTIONARY_FILE: {self.TOKEN_DICTIONARY_FILE}")
        print(f"ENSEMBL_MAPPING_FILE: {self.ENSEMBL_MAPPING_FILE}")
        print(f"Output directory: {self.output_dir}")
        print(f"Classifier type: {self.classifier_type}")

    def _get_pretrained_model_path(self):
        """
        Get the path to the pretrained model, preferring final trained model over checkpoints.
        
        Returns:
        --------
        str
            Path to the pretrained model directory
            
        Raises:
        -------
        FileNotFoundError
            If no trained model or checkpoints are found
        """
        
        # Base path
        pretrained_model_path = os.path.join(
            self.BASE_DIR, "trained_foundation_models", "models", 
            f"30M_AGG{self.AGGREGATION_LEVEL}_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw"
        )
        
        # Check if final_trained_model subfolder exists
        final_model_path = os.path.join(pretrained_model_path, "final_trained_model", self.AGGREGATION_LEVEL)
        final_model_path_empty = False
        
        if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
            # Check if the folder has model files (not just empty)
            model_files = glob.glob(os.path.join(final_model_path, "*.bin")) + \
                            glob.glob(os.path.join(final_model_path, "*.safetensors")) + \
                            glob.glob(os.path.join(final_model_path, "config.json"))
            
            if model_files:
                pretrained_model_path = final_model_path
                print(f"✓ Using final trained model: {pretrained_model_path}")
            else:
                print(f"⚠ Final model folder exists but appears empty: {final_model_path}")
                final_model_path_empty = True
                # Fall through to checkpoint search
        else:
            print(f"ℹ Final model folder not found: {final_model_path}")
        
        # If final model not found or empty, find most recent checkpoint
        if not (os.path.exists(final_model_path)) or final_model_path_empty:
            # Find all checkpoint folders
            checkpoint_pattern = os.path.join(pretrained_model_path, "checkpoint-*")
            checkpoint_folders = glob.glob(checkpoint_pattern)
            
            if checkpoint_folders:
                # Extract checkpoint numbers and find the highest one
                checkpoint_numbers = []
                for folder in checkpoint_folders:
                    folder_name = os.path.basename(folder)
                    if folder_name.startswith("checkpoint-"):
                        try:
                            checkpoint_num = int(folder_name.split("-")[1])
                            checkpoint_numbers.append((checkpoint_num, folder))
                        except ValueError:
                            continue
                
                if checkpoint_numbers:
                    # Sort by checkpoint number and get the highest one
                    most_recent_checkpoint = max(checkpoint_numbers, key=lambda x: x[0])
                    pretrained_model_path = most_recent_checkpoint[1]
                    print(f"✓ Using most recent checkpoint: {pretrained_model_path} (step {most_recent_checkpoint[0]})")
                else:
                    print(f"❌ No valid checkpoint folders found in: {pretrained_model_path}")
                    raise FileNotFoundError(f"No trained model or checkpoints found in {pretrained_model_path}")
            else:
                print(f"❌ No checkpoint folders found in: {pretrained_model_path}")
                raise FileNotFoundError(f"No trained model or checkpoints found in {pretrained_model_path}")
        
        print(f"Final pretrained model path: {pretrained_model_path}")
        return pretrained_model_path
    
    def finetune_model(self, training_args, cell_state_dict, filter_data_dict, 
                  input_data_file, output_prefix, train_test_id_split_dict, 
                  train_valid_id_split_dict, num_crossval_splits=1,
                  freeze_num_encoder_layers=2, freeze_entire_model=False):
        """
        Fine-tune the model using the Classifier.
        
        Parameters:
        -----------
        training_args : dict
            Training arguments dictionary containing hyperparameters
        cell_state_dict : dict
            Cell state dictionary for classification
        filter_data_dict : dict
            Data filtering configuration
        input_data_file : str
            Path to input data file
        output_prefix : str
            Prefix for output files
        train_test_id_split_dict : dict
            Dictionary for train/test split IDs
        train_valid_id_split_dict : dict
            Dictionary for train/validation split IDs
            
        Returns:
        --------
        dict
            All metrics from validation
        """
        
        # Get pretrained model path
        pretrained_model_path = self._get_pretrained_model_path()
        
        # Initialize Classifier
        # OF NOTE: token_dictionary_file must be set to the gc-30M token dictionary if using a 30M series model
        # (otherwise the Classifier will use the current default model dictionary)
        # 30M token dictionary: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
        cc = Classifier(
            classifier=self.classifier_type,
            # model_version=self.model_version,
            cell_state_dict=cell_state_dict,
            filter_data=filter_data_dict,
            training_args=training_args,
            max_ncells=None,
            freeze_layers=freeze_num_encoder_layers,
            num_crossval_splits=num_crossval_splits,
            forward_batch_size=200,
            token_dictionary_file=self.TOKEN_DICTIONARY_FILE,
            nproc=16,
            ngpu = self.num_gpus,
            freeze_entire_model = freeze_entire_model
        )
        
        ### num_crossval_splits : {0, 1, 5}
        #     | 0: train on all data without splitting
        #     | 1: split data into train and eval sets by designated split_sizes["valid"]
        #     | 5: split data into 5 folds of train and eval sets by designated split_sizes["valid"]
        # split_sizes : None, dict
        #     | Dictionary of proportion of data to hold out for train, validation, and test sets
        #     | {"train": 0.8, "valid": 0.1, "test": 0.1} if intending 80/10/10 train/valid/test split
        
        # Prepare the data for training
        # Example input_data_file for 30M model: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
        cc.prepare_data(
            input_data_file=input_data_file,
            output_directory=self.output_dir,
            output_prefix=output_prefix,
            split_id_dict=train_test_id_split_dict
        )

        if num_crossval_splits==1:
        
            prepared_input_data_file = f"{self.output_dir}/{output_prefix}_labeled_train.dataset"
        elif num_crossval_splits==5:
            prepared_input_data_file = f"{self.output_dir}/{output_prefix}_labeled.dataset"
        else:
            raise ValueError("num_crossval_splits must be either 1 or 5")
        
        # Example 6 layer 30M Geneformer model: https://huggingface.co/ctheodoris/Geneformer/blob/main/gf-6L-30M-i2048/model.safetensors
        all_metrics = cc.validate(
            model_directory=pretrained_model_path,
            prepared_input_data_file=prepared_input_data_file,
            id_class_dict_file=f"{self.output_dir}/{output_prefix}_id_class_dict.pkl",
            output_directory=self.output_dir,
            output_prefix=output_prefix,
            split_id_dict=train_valid_id_split_dict
            # to optimize hyperparameters, set n_hyperopt_trials=100 (or alternative desired # of trials)
        )
        
        return all_metrics
            

    


# Example usage:
if __name__ == "__main__":

    task_config = {
            "tasks": {
            "disease_classification": [
                "genecorpus_heart_disease", 
                "cellnexus_blood_disease", 
                "cellnexus_covid_disease"
                ],
            # "dosage_sensitivity": ["genecorpus_dosage_sensitivity"]
            },
            "aggregation_levels": [
                # "metacell_2", 
                # "metacell_4", 
                # "metacell_8", 
                # "metacell_16", 
                "metacell_32", 
                "metacell_64", 
                "metacell_128"
                ]
        }

    # aggregation_level="metacell_32"
    # task="dosage_sensitivity"
    # dataset="genecorpus_dosage_sensitivity"
    model_version="V1"
    base_dir= "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer"
    model_variant="30M"
    crossval_splits = 1 # 1 or 5
    freeze_num_encoder_layers=6
    freeze_entire_model=True

        # Loop over tasks
    for task, datasets in task_config["tasks"].items():
        # Loop over each dataset associated with the task
        for dataset in datasets:
            # Loop over aggregation levels
            for aggregation_level in task_config["aggregation_levels"]:
                print(f"Running task={task}, dataset={dataset}, aggregation_level={aggregation_level}")

                for crossval_split in range(1, crossval_splits + 1):

                    crossval_split_metrics = {}



                    training_args = {
                    "num_train_epochs": 10,
                    "learning_rate": 0.000804,
                    "lr_scheduler_type": "polynomial",
                    "warmup_steps": 1812,
                    "weight_decay":0.258828,
                    "per_device_train_batch_size": 128,
                    "seed": 73,
                    "evaluation_strategy":"epoch",        # Evaluate every epoch
                    "save_strategy":"epoch",              # Save checkpoint every epoch
                    "metric_for_best_model":"eval_loss",  # Metric to determine "best" model # Doc: https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html#:~:text=after%20each%20evaluation.-,metric_for_best_model,-(str%2C
                    "greater_is_better":False,            # For loss, lower is better
                    "load_best_model_at_end":True,        # KEY: Load best model at the end
                    "save_total_limit":3,                 # Keep only 3 best checkpoints
                    # "logging_dir": os.path.normpath("D:/geneformer_finetuning/trained_cell_classification_models/disease_classification/genecorpus_heart_disease/30M_metacell_8/250623_geneformer_cellClassifier_genecorpus_heart_disease_test/ksplit1/runs"),
                    }
                    

                    input_data_file, cell_state_dict, filter_data_dict, train_test_id_split_dict, train_valid_id_split_dict = get_train_valid_test_splits(
                        task=task,
                        dataset=dataset,
                        model_variant=model_variant,
                        dataset_path=os.path.join(base_dir, "datasets"),
                        crossval_splits=crossval_splits,
                    )

                    finetuner = FineTuner(base_dir=base_dir,
                        aggregation_level=aggregation_level,
                        model_variant=model_variant,
                        task=task,
                        dataset=dataset)
                    
                    output_prefix = "test" if crossval_splits == 1 else "ksplit" + str(crossval_split)
        
                    all_metrics = finetuner.finetune_model(
                        training_args=training_args, 
                        cell_state_dict = cell_state_dict, 
                        filter_data_dict = filter_data_dict, 
                        input_data_file = input_data_file, 
                        output_prefix = output_prefix, 
                        train_test_id_split_dict = train_test_id_split_dict, 
                        train_valid_id_split_dict = train_valid_id_split_dict, 
                        num_crossval_splits=1 if task != "dosage_sensitivity" else crossval_splits,
                        freeze_num_encoder_layers=freeze_num_encoder_layers,
                        freeze_entire_model=freeze_entire_model
                        )
                    
                    crossval_split_metrics[str(crossval_split)] = all_metrics

                metrics_path = os.path.join(finetuner.output_dir, "metrics.json")

                # Save the metrics to that path
                with open(metrics_path, "w") as f:
                    json.dump(crossval_split_metrics, f, indent=4)
    