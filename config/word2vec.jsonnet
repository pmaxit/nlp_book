local embedding_dim = 300;
local batch_size = 128;
local cuda_device = -1;

{"dataset_reader":{
    "type": "skip_gram",
    "load_from_pkl": "./junks/word_table.pkl"
    },
    "train_data_path":"./data/text8",
    "model":{
        "type": "skipgram_simple3",
        "embedding_in":{
            "embedding_dim": embedding_dim,
            "vocab_namespace": "tags_in",
            "trainable": true
            },
           "embedding_out":{
                "embedding_dim": embedding_dim,
                "vocab_namespace": "tags_in",
                "trainable": true
            },
        "cuda_device": cuda_device,
    },
    "data_loader":{
        "batch_size": batch_size,
        "shuffle":true,
        "max_instances_in_memory": batch_size*100

    },
    "trainer":{
        "num_epochs": 6,
        "patience": 3,
        "cuda_device": cuda_device,
        "grad_clipping": 5.0,
        "validation_metric": "+f1",
        "run_sanity_checks":false,
        "num_gradient_accumulation_steps":1,
        "optimizer":{
            "type": "sgd",
            "lr": 0.001,
            "momentum": 0.99
        },
    },
}