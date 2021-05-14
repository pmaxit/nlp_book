local EMBEDDING_DIM=128;
local RNN_DIM=256;
local RNN_NUM_LAYERS = 2;
local BEAM_SIZE = 4;
local LR = 0.001;
local CUDA_DEVICE= 0;
local vocab_size = 50000;
local SOURCE_MAX_TOKENS= 400;
local TARGET_MAX_TOKENS=100;
local VOCAB_SIZE = 50000;
local NUM_EPOCHS = 4;
local BATCH_SIZE = 32;
local MAX_DECODING_STEPS=100;
{
    "dataset_reader":{
        "type":"cnn_dailymail",
        "tokenizer": {
            "type":"whitespace"
        },
        "target_namespace": "target_tokens",
        "lowercase": true,
        "source_max_tokens": SOURCE_MAX_TOKENS,
        "target_max_tokens": TARGET_MAX_TOKENS,
        "seperate_namespaces": true,
        "save_pgn_fields": true,
        //"max_instances": 1000
    },
    "vocabulary":{
        "max_vocab_size": VOCAB_SIZE
    },
    "train_data_path": "train",
    "validation_data_path": "valid",
    "model":{
        "type":"pgn",
        "target_namespace":"target_tokens",
        "embed_attn_to_output": true,
        "source_embedder":{
            "type":"basic",
            "token_embedders":{
                "tokens":{
                    "type": "embedding",
                    "embedding_dim" : EMBEDDING_DIM
                },
            },
        },
        "encoder":{
            "type" : "lstm",
            "num_layers": RNN_NUM_LAYERS,
            "input_size": EMBEDDING_DIM,
            "hidden_size": RNN_DIM,
            "bidirectional": true
        },
        "attention": {
            "type": "dot_product"
        },
    "use_coverage": false,
    "coverage_loss_weight": 0.0,
    "max_decoding_steps": MAX_DECODING_STEPS,
    "beam_size": BEAM_SIZE
   },
   "data_loader":{
       "batch_size": BATCH_SIZE,
       "max_instances_in_memory": BATCH_SIZE*100,
   },
    "trainer":{
        "num_epochs": NUM_EPOCHS,
        "grad_norm":2.0,
        "cuda_device": CUDA_DEVICE,
        "patience": 1,
        "optimizer":{
            "type":"adam",
            "lr": LR
        },
    },
}