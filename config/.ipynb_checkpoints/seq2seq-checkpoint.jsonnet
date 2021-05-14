{"dataset_reader":{
    "type": "seq2seq_rev",
    "source_tokenizer": {
        "type": "whitespace"
    },
    "target_tokenizer":{
        "type": "whitespace"
    },
},
"train_data_path":"data/reverse/train.csv",
"validation_data_path": "data/reverse/dev.csv",
"test_data_path": "data/reverse/test.csv",
"model":{
    "type": "simple_seq2seq",
    "max_decoding_steps": 30,
    "use_bleu": true,
    "beam_size": 1,
    "attention":{
        "type": "bilinear",
        "vector_dim": 128,
        "matrix_dim": 128
    },
    "source_embedder":{
        "token_embedders":{
            "tokens":{
                "type": "embedding",
                "embedding_dim": 16
            },
        },
    },
    "encoder":{
        "type": "lstm",
        "input_size": 16,
        "hidden_size": 64,
        "bidirectional": true,
        "num_layers": 1,
        "dropout": 0.1
    },
},
"data_loader":{
    "batch_size":32
},
"trainer":{
    "cuda_device":-1,
    "num_epochs": 10,
    "learning_rate_scheduler":{
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 5
    },
    "optimizer":{
        "lr": 0.001,
        "type": "adam"
    },
}
}