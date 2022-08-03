"# FloodEmergencyDecision_QA" 洪涝灾害应急决策自动问答系统，使用UER_py训练模型https://github.com/dbiir/UER-py

数据：datasets文件夹，包括防洪应急预案，应急决策问答对

模型：GPT2中文预训练模型，https://huggingface.co/uer/gpt2-chinese-cluecorpussmall

使用说明：

1. 下载项目

2. 下载预训练模型，https://huggingface.co/uer/gpt2-chinese-cluecorpussmall

3. 模型转换
python scripts/convert_gpt2_from_huggingface_to_uer.py --input_model_path "D:/gpt2-chinese-cluecorpussmall/pytorch_model.bin" --output_model_path models/gpt2_pytorch_model.bin --layers_num 12

4. 增量预训练

    4.1 数据处理
    
    /uer/util/data.py 修改class LmDataset(Dataset)，修改内容见注释
    
    python preprocess.py --corpus_path "/datasets/train-un.txt" --vocab_path models/google_zh_vocab.txt --dataset_path train-un.pt --seq_length 1024 --target lm
    
    4.2 模型训练
    
    uer/train.py 修改class Trainer(object)，修改def worker(proc_id, gpu_ranks, args, model)，修改内容见注释
    
    python pretrain.py --dataset_path train-un.pt --pretrained_model_path models/gpt2_pytorch_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/gpt2/config.json --output_model_path models/train-un-1024gpt2_model.bin --world_size 1 --gpu_ranks 0 --total_steps 10000 --save_checkpoint_steps 5000 --report_steps 1000 --learning_rate 5e-5 --batch_size 8 --embedding word_pos --remove_embedding_layernorm --encoder transformer --mask causal --layernorm_positioning pre  --target lm --tie_weights
    
5 模型训练
    
    4.1 数据处理
    
    python preprocess.py --corpus_path "/datasets/train-96.txt" --vocab_path models/google_zh_vocab.txt --dataset_path train-96.pt --seq_length 256 --target lm
    
    4.2 模型训练
    
    python pretrain.py --dataset_path train-96.pt --dataset_valid_path valid-32.pt --pretrained_model_path models/train-un-1024_gpt2_model.bin-10000 --vocab_path models/google_zh_vocab.txt --config_path models/gpt2/config.json --output_model_path models/train-un-96-gpt2_model.bin --world_size 1 --gpu_ranks 0 --total_steps 10000 --save_checkpoint_steps 5000 --report_steps 1000 --learning_rate 5e-5 --batch_size 32 --embedding word_pos --remove_embedding_layernorm --encoder transformer --mask causal --layernorm_positioning pre  --target lm --tie_weights
    
    4.3 模型测试
    
    python scripts/generate_lm_deepspeed.py --deepspeed --deepspeed_config models/deepspeed_config.json --load_model_path models/train-un-96-gpt2_model.bin-10000 --vocab_path models/google_zh_vocab.txt --test_path test-32.txt --prediction_path test-un-32-output.txt --config_path models/gpt2/config.json --seq_length 256 --embedding word_pos --remove_embedding_layernorm --encoder transformer --mask causal --layernorm_positioning pre --target lm --tie_weights

