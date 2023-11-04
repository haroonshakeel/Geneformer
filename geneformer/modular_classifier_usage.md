# Cell classifier
def finetune_cells(token_set = Path('geneformer/token_dictionary.pkl'), median_set = Path('geneformer/gene_median_dictionary.pkl'), pretrained_model = ".",
 dataset = 'Genecorpus-30M/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/',
 dataset_split = None,
  filter_cells = .005,
  epochs = 1,
  cpu_cores = os.cpu_count(),
  geneformer_batch_size = 12,
  optimizer = 'adamw',
  max_lr = 5e-5,
  num_gpus = torch.cuda.device_count(),
  max_input_size = 2 ** 11,
  lr_schedule_fn = "linear",
  warmup_steps = 500,
  freeze_layers = 0,
  emb_extract = False,
  max_cells = 1000,
  emb_layer = 0,
  emb_filter = None,
  emb_dir = 'embeddings',
  overwrite = True,
  label = "cell_type",
  data_filter = None,
  forward_batch = 200, model_location = None,
  skip_training = False,
  sample_data = 1,
   inference = False,
   optimize_hyperparameters = False,
   output_dir = None):

    '''
    Primary Parameters
    -------------------
    dataset: path
        Path to fine-tuning/testing dataset for training

    model_location: path
        Path to location of existing model to use for inference and embedding extraction

    pretrained_model: path
        Path to pretrained GeneFormer 30M model before fine-tuning

    inference: bool
        Chooses whether to perform inference (which causes the function to return the list of similarities). Defaults to False

    skip_training: bool
        Chooses whether to skip training the model. Defaults to False

    emb_extract: bool
        Choose whether to extract embeddings and calculate similarities. Defaults to True

    optimize_hyperparameters: bool
        Choose whether to optimize model hyperparamters. Defaults to False
    label: string
		The label string in the formatted dataset that contains true class labels. Defaults to "label"

    Customization Parameters
    -------------------

    dataset_split: str
        How the dataset should be partitioned (if at all), and what ID should be used for partitioning

    data_filter: list
        (For embeddings and inference) Runs analysis subsets of the dataset by the ID defined by dataset_split

    label: str
        What feature should be read as a classification label

    emb_layer: int
        What layer embeddings should be extracted and compared from.

    emb_filter: ['cell1', 'cell2'...]
        Allows user to narrow down range of cells that embeddings will be extracted from.

    max_cells: int
        How many embeddings from cells should be extracted.

    freeze_layers: int
        Number of layers should be permanently frozen during fine-tuning (starting from the first layer, 4 brings it up to the pretrained model).

    sample_data: float
        What proportion of the HF dataset should be used

    '''

   # Gene Classifier
   def classify_genes(gene_info = "Genecorpus-30M/example_input_files/gene_info_table.csv",
   genes = "Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sens_tf_labels.csv",
  corpus_30M = "Genecorpus-30M/genecorpus_30M_2048.dataset/", model = '.',
  max_input_size = 2 ** 11,
  max_lr = 5e-5,
  freeze_layers = 4,
  num_gpus = 1,
  num_proc = os.cpu_count(),
  geneformer_batch_size = 9,
  epochs = 1,
  filter_dataset = 50_000,
  emb_extract = True,
  emb_layer = 0,
  forward_batch = 200,
  filter_data = None,
  inference = False,
  k_validate = True,
  model_location = "230917_geneformer_GeneClassifier_dosageTF_L2048_B12_LR5e-05_LSlinear_WU500_E1_Oadamw_n10000_F4/",
  skip_training = False,
  emb_dir = 'gene_emb',
  output_dir = None,
  max_cells = 1000,
  num_cpus = os.cpu_count()):

    """"
    Primary Parameters
    -----------

    gene_info: path
        Path to gene mappings

    corpus_30M: path
        Path to 30M Gene Corpus

    model: path
        Path to pretrained GeneFormer model

    genes: path
        Path to csv file containing different columns of genes and the column labels

    inference: bool
        Whether the model should be used to run inference. If False, model will train with labeled data instead. Defaults to False

    k_validate: bool
        Whether the model should run k-fold validation or simply perform regular training/evaluate. Defaults to True

    skip_training: bool
        Whether the model should skip the training portion. Defaults to False

    emb_extract: bool
        WHether the model should extract embeddings for a given gene (WIP)


    Customization Parameters
    -----------

    freeze_layers: int
        Freezes x number of layers from the model. Default is 4 (2 non-frozen layers)

    filter_dataset: int
        Number of cells to filter from 30M dataset. Default is 50_000

    emb_layer: int
        What layer embeddings are extracted from. Default is 4

    filter_data: str, list
        Filters down embeddings to a single category. Default is None


    """
