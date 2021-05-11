local fasttext_embedding_dim = 300;
local glove_embedding_dim = 100;
local lstm_hidden_size = 100;

local batch_size = 128;
local dropout = 0.3;
local max_length = 1024;

{"dataset_reader":{
    "type": "toxic",
    "max_length": max_length,
    "token_indexers":{
       "tokens1":{
           "type": "single_id"
       },
       "tokens2":{
           "type": "single_id"
       },
    }
},
"train_data_path": "data/toxic/train.csv",
"validation_data_path": "data/toxic/valid.csv",
//"test_data_path": "data/toxic/test.csv",
//"vocabulary":{
//    "type": "from_files", 
//    "directory": "/notebooks/nlp_book/junks/vocab.tar.gz"
//},
"model":{
    "type": "toxic",
    "text_field_embedder":{
        "token_embedders":{
            "tokens1":{
                "type": "embedding",
                "pretrained_file": "/notebooks/nlp_book/junks/crawl-300d-2M/crawl-300d-2M.vec",
                "embedding_dim": fasttext_embedding_dim,
                "trainable": false
            },
            "tokens2": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "embedding_dim": glove_embedding_dim,
                    "trainable": false
                }
        },
    },
    "encoder":{
            "type": "cnn",
            "embedding_dim": fasttext_embedding_dim + glove_embedding_dim,
            "num_filters": 32
    },
    "classifier_feedforward":{
        "input_dim": 32*4,
        "num_layers": 2,
        "hidden_dims": [200, 6],
        "activations": ["tanh", "linear"],
        "dropout": [0.2, 0.0],
    }
},
"data_loader":{
    "batch_size": batch_size,
    //"max_instances_in_memory": batch_size*100,
},
"trainer":{
    "num_epochs": 40,
    "patience": 3,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+f1",
    "num_gradient_accumulation_steps":2,
    "optimizer":{
        "type": "adadelta"
    },
},
}

