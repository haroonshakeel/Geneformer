import os
import json


def get_train_valid_test_splits(TASK, DATASET, MODEL_VARIANT, DATASET_PATH, CROSSVAL_SPLITS=1):
    """
    Get train, validation, and test splits based on the task and dataset.

    Parameters:
    - TASK: str, type of task (e.g., "disease_classification", "dosage_sensitivity")
    - DATASET: str, name of the dataset
    - MODEL_VARIANT: str, model variant (e.g., "30M", "95M")
    - DATASET_PATH: str, path to the dataset directory

    Returns:
    - input_data_file: str, path to the input data file
    - cell_state_dict: dict, dictionary containing cell state information
    - filter_data_dict: dict, dictionary for filtering data
    - train_test_id_split_dict: dict, dictionary for train-test split
    - train_valid_id_split_dict: dict, dictionary for train-validation split
    """
    if TASK == "disease_classification" and DATASET == "genecorpus_heart_disease":
        input_data_file = os.path.join(DATASET_PATH, "human_dcm_hcm_nf.dataset")
        cell_state_dict = {"state_key": "disease", "states": "all"}
        filter_data_dict={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]}

        # previously balanced splits with prepare_data and validate functions
        # argument attr_to_split set to "individual" and attr_to_balance set to ["disease","lvef","age","sex","length"]
        train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371", "1549", "1515"]
        eval_ids = ["1422", "1510", "1539", "1606", "1702"]
        test_ids = ["1437", "1516", "1602", "1685", "1718"]
        
        train_test_id_split_dict = {"attr_key": "individual",
                                    "train": train_ids+eval_ids,
                                    "test": test_ids}

        train_valid_id_split_dict = {"attr_key": "individual",
                                "train": train_ids,
                                "eval": eval_ids}
    elif TASK == "disease_classification" and DATASET == "cellnexus_blood_disease":
        input_data_file = os.path.join(DATASET_PATH, "tokenized_" + str(MODEL_VARIANT), "cellnexus_singlecell.dataset")
        filter_data_dict = None
        cell_state_dict = {"state_key": "disease", "states": "all"}


        # Load train_test split dictionary
        train_test_file = os.path.join(DATASET_PATH, "train_test_id_split_dict.json")
        print(f"Loading train_test split from: {train_test_file}")

        with open(train_test_file, 'r') as f:
            train_test_id_split_dict = json.load(f)

        # Load train_valid split dictionary  
        train_valid_file = os.path.join(DATASET_PATH, "train_valid_id_split_dict.json")
        print(f"Loading train_valid split from: {train_valid_file}")

        with open(train_valid_file, 'r') as f:
            train_valid_id_split_dict = json.load(f)

        # Verify the loaded dictionaries
        print("\n=== Train-Test Split Dictionary ===")
        print(f"Attribute key: {train_test_id_split_dict['attr_key']}")
        print(f"Train samples: {len(train_test_id_split_dict['train'])}")
        print(f"Test samples: {len(train_test_id_split_dict['test'])}")

        print("\n=== Train-Valid Split Dictionary ===")
        print(f"Attribute key: {train_valid_id_split_dict['attr_key']}")
        print(f"Train samples: {len(train_valid_id_split_dict['train'])}")
        print(f"Eval samples: {len(train_valid_id_split_dict['eval'])}")

        # Show a few sample IDs for verification
        print(f"\nSample train IDs: {train_test_id_split_dict['train'][:5]}")
        print(f"Sample test IDs: {train_test_id_split_dict['test'][:5]}")
        print(f"Sample eval IDs: {train_valid_id_split_dict['eval'][:5]}")

        print("\n✅ Dictionaries loaded successfully!")
            # previously balanced splits with prepare_data and validate functions
            # argument attr_to_split set to "individual" and attr_to_balance set to ["disease","
    elif TASK == "disease_classification" and DATASET == "cellnexus_covid_disease":
        input_data_file = os.path.join(DATASET_PATH, "tokenized_" + str(MODEL_VARIANT), "cellnexus_singlecell.dataset")
        filter_data_dict = None
        cell_state_dict = {"state_key": "disease", "states": "all"}


        # Load train_test split dictionary
        train_test_file = os.path.join(DATASET_PATH, "train_test_id_split_dict.json")
        print(f"Loading train_test split from: {train_test_file}")

        with open(train_test_file, 'r') as f:
            train_test_id_split_dict = json.load(f)

        # Load train_valid split dictionary  
        train_valid_file = os.path.join(DATASET_PATH, "train_valid_id_split_dict.json")
        print(f"Loading train_valid split from: {train_valid_file}")

        with open(train_valid_file, 'r') as f:
            train_valid_id_split_dict = json.load(f)

        # Verify the loaded dictionaries
        print("\n=== Train-Test Split Dictionary ===")
        print(f"Attribute key: {train_test_id_split_dict['attr_key']}")
        print(f"Train samples: {len(train_test_id_split_dict['train'])}")
        print(f"Test samples: {len(train_test_id_split_dict['test'])}")

        print("\n=== Train-Valid Split Dictionary ===")
        print(f"Attribute key: {train_valid_id_split_dict['attr_key']}")
        print(f"Train samples: {len(train_valid_id_split_dict['train'])}")
        print(f"Eval samples: {len(train_valid_id_split_dict['eval'])}")

        # Show a few sample IDs for verification
        print(f"\nSample train IDs: {train_test_id_split_dict['train'][:5]}")
        print(f"Sample test IDs: {train_test_id_split_dict['test'][:5]}")
        print(f"Sample eval IDs: {train_valid_id_split_dict['eval'][:5]}")

        print("\n✅ Dictionaries loaded successfully!")
    return input_data_file, cell_state_dict, filter_data_dict, train_test_id_split_dict, train_valid_id_split_dict