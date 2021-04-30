{"dataset_reader":{
    "type": "toxic",
    "max_length": 5000
},
"train_data_path": "data/toxic/train.csv",
"validation_data_path": "data/toxic/val.csv",
"test_data_path": "data/toxic/test.csv",
"model":{
    "type": "toxic",
    "text_field_embedder":{
        "token_embedders":{
            "tokens":{
                "type": "embedding",
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
                "trainable": false
            },
        },
    },
    "encoder":{
        "type":"lstm",
        "bidirectional": true,
        "input_size": 100,
        "hidden_size": 100,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "classifier_feedforward":{
        "input_dim": 200,
        "num_layers": 2,
        "hidden_dims": [200, 6],
        "activations": ["tanh", "linear"],
        "dropout": [0.2, 0.0],
    }
}
,
"data_loader":{
    "batch_size": 64
},
"trainer":{
    "num_epochs": 40,
    "patience": 3,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+f1",
    "optimizer":{
        "type": "adagrad"
    },
},
}