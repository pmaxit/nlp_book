local fasttext_embedding_dim = 300;
local glove_embedding_dim = 200;
local lstm_hidden_size = 40;

local batch_size = 32;
local dropout = 0.3;
local max_length = 1024;

{"dataset_reader":{
    "type": "toxic",
    "max_length": max_length,
    "token_indexers":{
       "tokens1":{
           "type": "single_id",
           "lowercase_tokens": true
       },
       "tokens2":{
           "type": "single_id",
           "lowercase_tokens": true
       },
    }
},
"train_data_path": "data/toxic/train.csv",
//"validation_data_path": "data/toxic/valid.csv",
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
                "pretrained_file": "input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec",
                "embedding_dim": fasttext_embedding_dim,
                "trainable": false
            },
            "tokens2": {
                    "type": "embedding",
                    "pretrained_file": "input/glove-stanford/glove.twitter.27B.200d.txt",
                    "embedding_dim": glove_embedding_dim,
                    "trainable": false
                }
        },
    },
    "encoder":{
            "type": "swem",
            "emebdding_dim": fasttext_embedding_dim + glove_embedding_dim
    },
    "classifier_feedforward":{
        "input_dim": lstm_hidden_size*2,
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
