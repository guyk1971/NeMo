#!/bin/bash

# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/gpt/gpt_training.html


python megatron_gpt_pretraining_gk.py  \
    --config-path=conf \
    --config-name=megatron_gpt_config \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.max_epochs=null \
    trainer.max_steps=300000 \
    trainer.val_check_interval=300 \
    trainer.log_every_n_steps=50 \
    trainer.limit_val_batches=50 \
    trainer.limit_test_batches=50 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=16 \
    model.micro_batch_size=6 \
    model.global_batch_size=192 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.max_position_embeddings=1024 \
    model.encoder_seq_length=1024 \
    model.hidden_size=768 \
    model.ffn_hidden_size=3072 \
    model.num_layers=12 \
    model.num_attention_heads=12 \
    model.init_method_std=0.021 \
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.library=sentencepiece \
    model.tokenizer.model=spm_32k_wiki.model \
    model.data.data_prefix=[1.0,gpt_training_data_text_document] \
    model.data.num_workers=2 \
    model.data.seq_length=1024 \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=fused_adam \
    model.optim.lr=6e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=750 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=6e-5 \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=val_loss \
    exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.always_save_nemo=False