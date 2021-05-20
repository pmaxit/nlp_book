local EMBEDDING_DIM=128;
local RNN_DIM=256;
local RNN_NUM_LAYERS = 2;
local BEAM_SIZE = 4;
local LR = 0.001;
local CUDA_DEVICE= -1;
local vocab_size = 50000;
local SOURCE_MAX_TOKENS= 400;
local TARGET_MAX_TOKENS=100;
local VOCAB_SIZE = 50000;
local NUM_EPOCHS = 4;
local BATCH_SIZE = 32;
local MAX_DECODING_STEPS=100;
local MODEL_NAME = "t5-small";
{
    "dataset_reader":{
        "type":"cnn_dailymail",
        "tokenizer": {
            "type":"pretrained_transformer",
            "model_name": MODEL_NAME
        },
        "source_token_indexers":{
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": MODEL_NAME,
                "namespace": "tokens",
            }
        },
        "target_namespace": "target_tokens",
        "lowercase": true,
        "source_max_tokens": SOURCE_MAX_TOKENS,
        "target_max_tokens": TARGET_MAX_TOKENS,
        "seperate_namespaces": false    ,
        "save_pgn_fields": false,
        "save_copy_fields": false,
        "source_prefix": "summarize: ",
        //"max_instances": 1000
    },
    "vocabulary":{
        "max_vocab_size": VOCAB_SIZE
    },
    "train_data_path": "https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary.csv",
    //"validation_data_path": "valid",
    "model":{
        "type":"t5",
        "model_name": MODEL_NAME
   },
   "data_loader":{
       "batch_size": BATCH_SIZE,
       "shuffle":true
       //"max_instances_in_memory": BATCH_SIZE*100,
   },
    "trainer":{
        "num_epochs": NUM_EPOCHS,
        "grad_norm":2.0,
        "cuda_device": CUDA_DEVICE,
        "patience": 1,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true,
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
    },
}