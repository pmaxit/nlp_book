{"dataset_reader":{
    "type": "seq2seq",
    "source_tokenizer": {
        "type": "character",
    },
    "start_symbol": "~",
    "end_symbol": "~"
},
"train_data_path":"data/reverse/train.in",
"validation_data_path": "data/reverse/val.in",
"test_data_path": "data/reverse/test.in",
"model":{
    "type": "simple_seq2seq",
    "max_decoding_steps": 30,
    "use_bleu": true,
    "beam_size": 1,
    // if using attention, a weighted average oveer encoder outputs
    // will be concated to the previous target embedding to form the input
    // to the decoder at each time step.

    // bilinear attention computes attention between a vector and a matrix using 
    // a bilinear attention function. This function has matrix of weights nd bias
    // X^T W y + b
    // 200 ( decoder) , (10 x 300)(encoder outputs)
    // 200 -> 1 x 300 (200, 300)
    // batch multiply . ( 1 x300) , (10X300) -> dot product
    // 10 x 1 attentions
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

    // vector dimension is set above because of encoder 
    // it started with 128 dimension and output decoder also had 128 dimension
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