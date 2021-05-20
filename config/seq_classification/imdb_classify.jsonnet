{
  "dataset_reader": {
    "type": "my_imdb",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
        "type": "whitespace"
    }
  },
  "train_data_path": "train",
  "test_data_path": "test",
  "evaluate_on_test": true,
  "model": {
    "type": "seq_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        }
      }
    },
    "seq2vec_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1
    },
    "feedforward": {
        "input_dim": 200,
        "num_layers": 2,
        "hidden_dims": [300,100],
        "activations": ["relu","relu"],
        "dropout": [0.35,0.15]    
    },
    "dropout": 0.2,
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
    "batch_size": 64
  },

  "trainer": {
    "num_epochs": 10,
    "grad_norm": 2.0,
    "cuda_device": -1,
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