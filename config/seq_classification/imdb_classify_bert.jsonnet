local batch_size = 32;
local dropout = 0.3;
local max_length = 512;
local bert_model = "distilbert-base-uncased";

{
  "dataset_reader": {
    "type": "my_imdb",
    
     "tokenizer":{
        "type":"pretrained_transformer",
        "model_name": bert_model,
         "max_length": max_length
    },
    "token_indexers":{
        "bert":{
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "max_length": max_length
        }
    }
  },
  "train_data_path": "train",
  "test_data_path": "test",
  "evaluate_on_test": true,
  "model": {
    "type": "rnn_classifier",
    "text_field_embedder": {
      "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name":bert_model,
                    "last_layer_only": true,
                    "max_length": max_length
                }
      }
    },
    "seq2vec_encoder": {
        "type": "cls_pooler",
        "embedding_dim": 768
    },
    "feedforward": {
        "input_dim": 200,
        "num_layers": 2,
        "hidden_dims": [300,100],
        "activations": ["relu","relu"],
        "dropout": [0.25]
    },
    "dropout": dropout,
      "regularizer": {
            "regexes": [
                [
                    ".*",
                    {
                        "type": "l2",
                        "alpha": 0.001
                    }
                ]
            ]
        }
  },
  "data_loader": {
    "shuffle":true,
    "batch_size": batch_size,
    "max_instances_in_memory": 1000,

  },

  "trainer": {
    "num_epochs": 10,
    "grad_norm": 2.0,
    "cuda_device": 0,
    "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
        },
    "moving_average": {
            "type": "exponential",
            "decay": 0.9999
    }
  }
}
