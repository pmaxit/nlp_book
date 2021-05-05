local batch_size = 32;
local dropout = 0.3;
local max_length = 512;

local bert_model = "bert-base-uncased";


{"dataset_reader":{
    "type": "toxic",
    "max_length": max_length,
    "tokenizer":{
        "type":"pretrained_transformer",
        "model_name": bert_model
    },
    "token_indexers":{
        "bert":{
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
        }
    },
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
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name":bert_model,
                    "last_layer_only": true,
                    "max_length": max_length
                }
        },
    },
    "encoder":{
	"type": "bert_pooler",
     "pretrained_model": bert_model,
     "requires_grad": false
    },
    "classifier_feedforward":{
        "input_dim": 768,
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
        "type": "adadelta",
        "lr" : 0.5,
        "rho": 0.95
    },
},
}
