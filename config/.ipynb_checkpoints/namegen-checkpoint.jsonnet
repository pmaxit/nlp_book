{"dataset_reader":{
    "type":"name_dataset"
   },
 "train_data_path": "./data/first_names.all.txt",
 "model": {
     "type": "namegen",
     "bidirectional": true,
     "sparse_embeddings": false,
     "text_field_embedder":{
         "token_embedders":{
             "tokens":{
                "type": "embedding",
                "embedding_dim": 16
                },
             },
         },
        "contextualizer":{
            "type": "lstm",
            "input_size": 16,
            "hidden_size": 64,
            "bidirectional": true,
            "num_layers": 1,
            "dropout": 0.1
        },
     },
 "data_loader":{
       "batch_size": 16
   }, 
 "trainer":{
       "num_epochs": 10,
       "optimizer":{
           "type": "adamw"
       }
   },
}