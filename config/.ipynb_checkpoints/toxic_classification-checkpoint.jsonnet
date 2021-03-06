local batch_size = 32;
local dropout = 0.3;
{"dataset_reader":{
    "type": "toxic",
    "max_length": 5000,
    "token_indexers":{
        "tokens":{
            'type': 'single_id',
            'lowercase_tokens': true
        },
        "tokens2":{
		    "type": "single_id",
		    "lowercase_tokens": true
	}
    },
},
"train_data_path": "data/toxic/train.csv",
"validation_data_path": "data/toxic/valid.csv",
//"test_data_path": "data/toxic/test.csv",
"vocabulary":{
    "type": "from_files", 
    "directory": "/notebooks/nlp_book/junks/vocab.tar.gz"
},
"model":{
    "type": "toxic",
    "text_field_embedder":{
        "token_embedders":{
            "tokens":{
                "type": "embedding",
                "pretrained_file": "./junks/crawl-300d-2M.vec",
                "embedding_dim": 300,
                "trainable": false
            },
	   "tokens2":{
		"type": "embedding",
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
	        "trainable": false
                
	    }
        },
    },
    "encoder":{
	"type": "bag_of_embeddings",
        "embedding_dim": 400
    },
    "classifier_feedforward":{
        "input_dim": 400,
        "num_layers": 2,
        "hidden_dims": [200, 6],
        "activations": ["tanh", "linear"],
        "dropout": [0.2, 0.0],
    }
}
,
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
        "lr" : 1.0,
        "rho": 0.95
    },
},
}
