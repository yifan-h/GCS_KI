## Graph Convolution Simulator (GCS)

### Requirements:

* Install all necessary libraries for [K-Adapter](https://github.com/microsoft/K-Adapter) and [ERNIE](https://github.com/thunlp/ERNIE).
* Prepare all data (pretraining data and knowledge graph data) and preprocess them. Users may download our pre-processed version [here]().

### Reproduce results in the paper:

#### 1. Run GCS to get attention coefficients for knowledge triples.

    sh run_gcs_ernie.sh
    sh run_gcs_kadapter.sh

You can run above two command lines for for ERNIE and K-Adapter models to get interpretation results. 

The results `0.1_all_attn.pt` would be saved in their data folder respectively (`./data/wikidata/results` for ERNIE and `./data/trex-rc/results` for K-Adapter). The model would also be saved as the file `model_gcs.pkl`.

#### 2. Drop KI corpus based on the interpretation results of GCS.

    sh run_kidrop_ernie.sh
    sh run_kidrop_kadapter.sh

You can run above two command lines for for ERNIE and K-Adapter models to get drop-UE KI corpus. 

For ERNIE, the subset of KI-corpus would be saved as `./data/wikidata/entity2vec.vec`. For K-Adapter, the subset of KI corpus would be saved as `./data/trex-rc/0.1_all_data_remove.json`.

Users may replace these files in ERNIE and K-Adapter to reproduce their KI processes.

#### 3. Remove test sets in downstream task based on the interpretation results of GCS.

    sh run_kidrop_ernie.sh
    sh run_kidrop_kadapter.sh
    
To get the performance of ERNIE and K-Adapter for different test sets in downstream task, users can run two above command lines. The results would be saved in the data folder. Users may replace the test set files in ERNIE and K-Adapter to reproduce their finetuning processes.

### For other experiments

Users may take a look at the `src/main.py` file, and read the comments for more details.
