#!/bin/bash

# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/gpt/gpt_training.html


python megatron_gpt_pretraining_gk.py  \
    --config-path=conf \
    --config-name=megatron_gpt_config \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.max_epochs=null \
    trainer.max_steps=100000 \
    trainer.val_check_interval=100 \
    trainer.log_every_n_steps=10 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16 \
    model.micro_batch_size=4 \
    model.global_batch_size=8 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.max_position_embeddings=512 \
    model.encoder_seq_length=512 \
    model.hidden_size=768 \
    model.ffn_hidden_size=3072 \
    model.num_layers=12 \
    model.num_attention_heads=12 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=gpt2-vocab.json \
    model.tokenizer.merge_file=gpt2-merges.txt \
    model.data.data_prefix=[1.0,hfbpe_gpt_training_data_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=512 \
    model.data.splits_string=\'900,50,50\' \
    model.optim.name=distributed_fused_adam \
    model.optim.lr=6e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=500 \
    model.optim.sched.constant_steps=50000 \
    model.optim.sched.min_lr=6e-5 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=False