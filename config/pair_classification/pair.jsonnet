{"dataset_reader":{
    "type": "pair_reader",
    "token_indexers":{
        "tokens":{
            "type":"single_id",
            "lowercase_tokens": true
        },
    },
    "tokenizer":{
        "type":"character"
    },
},
"train_data_path":"/Users/puneet/Projects/pytorch/practice/book_nlp/data/nicknames_20210415.txt",
"model":{
    "type": "pair_classify",
    "text_field_embedder":{
        "token_embedders":{
            "tokens":{
                "type": "embedding",
                "projection_dim": 10,
                "embedding_dim": 10,
                "trainable": true
            },
        },
    },
    "attend_feedforward":{ // projected dimension is 50
        "input_dim": 10,
        "num_layers": 1,
        "hidden_dims": 10,
        "activations": "relu",
        "dropout": 0.2
    },
    "matrix_attention": {"type": "dot_product"},

    "compare_feedforward": {
      "input_dim": 20,     // 50 own dimension and 50 the other pair word
      "num_layers": 1,
      "hidden_dims": 10,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 20,      // average of premise compared vector + average of hypothesis compare vector
      "num_layers": 1,
      "hidden_dims": 1,
      "activations": ["linear"],
    },
     "initializer": {
       "regexes": [
         [".*linear_layers.*weight", {"type": "xavier_normal"}],
         [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
       ]
     }
},
"data_loader":{
    "batch_size":32,
    "shuffle":true
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
    "validation_metric": "-loss",
    "optimizer":{
        "lr": 0.01,
        "type": "adam"
    },
 }
}