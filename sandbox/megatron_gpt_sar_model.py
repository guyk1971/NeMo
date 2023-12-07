# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split,average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.text_generation_utils import generate, get_computeprob_response
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronGPTSARModel']


class MegatronGPTSARModel(MegatronGPTModel):
    """
    Megatron GPT for Synthetic Associative Recall test
    the only added functionality is to add the accuracy_ignore_index metric in the validation step
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name, self.val_metric_kwargs = self.setup_metric(self.cfg.data.validation_ds)
            # self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.validation_ds, "metric"):
                self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')

        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name,self.test_metric_kwargs = self.setup_metric(self.cfg.data.test_ds)
            # self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.test_ds, "metric"):
                self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')


        self._validation_step_acc = None
        # self._test_step_acc = None

    # TODO SAR: make sure you focus on the accuracy with ignore_index
    def setup_metric(self, data_cfg):
        metric_name = "accuracy"
        metric_kwargs={'average':'micro'}
        inf_cfg={'compute_logprob':True}
        self.set_inference_config(inference_config=inf_cfg)
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric[metric_name]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                metric_kwargs['ignore_index']=data_cfg.metric.ignore_index
                metric_kwargs['task']=data_cfg.metric.task
                metric_kwargs['average']=data_cfg.metric.average
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
                
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric = MetricStringToTorchMetric[metric_name]
            
            # what are the file names ? can I delete this ?
            # if isinstance(data_cfg.file_names, ListConfig):
            #     if 'rouge' not in data_cfg.metric.name:
            #         metric = [
            #             metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
            #             for _ in range(len(data_cfg.file_names))
            #         ]
            #     else:
            #         metric = [metric() for _ in range(len(data_cfg.file_names))]
            # else:
            #     if 'rouge' not in data_cfg.metric.name:
            #         metric = [metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
            #     else:
            #         metric = [metric()]
            metric = metric(num_classes = self.tokenizer.vocab_size,**metric_kwargs).to(self.device)
        return metric, metric_name, metric_kwargs

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])


    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = next(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'loss_mask': batch['loss_mask'],
            }

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')
            output_tensor = model(**forward_args)
            
            if validation_step:
                logits=self.model(batch['tokens'],position_ids=batch['position_ids'],attention_mask=batch['attention_mask'])
                logits=logits[:,:,:self.tokenizer.vocab_size]
                preds=torch.argmax(logits,dim=-1)
                y=torch.full_like(batch['labels'],self.val_metric_kwargs['ignore_index'])
                y[:,-2]=batch['labels'][:,-2]
                acc=self.val_metric(preds,y)
                self.acc_per_micro_batch.append(acc)


            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func
    

    def validation_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Check if iterator is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            return
        mode = 'test' if self.trainer.testing else 'val'
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()
        self.acc_per_micro_batch=[]
        loss = self.fwd_bwd_step(dataloader_iter, batch_idx, True)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        self.validation_step_outputs.append(loss) if mode == 'val' else self.test_step_outputs.append(loss)

        self.validation_step_acc.append(torch.concat([a.unsqueeze(0) for a in self.acc_per_micro_batch]).mean())
        
        return loss


    def on_validation_epoch_end(self):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.cfg.data.get('validation_drop_last', True):
                averaged_loss = torch.stack(self.validation_step_outputs).mean()
                averaged_acc=torch.stack(self.validation_step_acc).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                total_loss_and_total_samples = torch.vstack(self.validation_step_outputs).sum(axis=0)
                avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                averaged_loss = avg_loss.type(torch.float32).cuda()
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        # When using pipeline parallelism, loss is calculated only in the last pipeline stage and
        # it should be casted to other pipeline stages for logging.
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if self.loss_broadcast_src_rank is None:
                dp_size = parallel_state.get_data_parallel_world_size()
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                pp_size = parallel_state.get_pipeline_model_parallel_world_size()
                rank_in_dp_tp_group = torch.distributed.get_rank() % (dp_size * tp_size)
                last_pipeline_stage_offset = (tp_size * dp_size) * (pp_size - 1)
                self.loss_broadcast_src_rank = last_pipeline_stage_offset + rank_in_dp_tp_group
            torch.distributed.broadcast(
                averaged_loss, self.loss_broadcast_src_rank, group=parallel_state.get_pipeline_model_parallel_group(),
            )

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        
        self.log('val_acc', averaged_acc, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_acc.clear()  # free memory


        return averaged_loss


    @property
    def validation_step_acc(self):
        """
        Cached outputs of validation_step. It can be a list of items (for single data loader) or a list of lists
        (for multiple data loaders).

        Returns:
            List of outputs of validation_step.
        """
        if self._validation_step_acc is not None:
            return self._validation_step_acc

        # Initialize new output list
        self._validation_step_acc = []
        # Check len(self._validation_dl) > 1 as sometimes single dataloader can be in a list: [<Dataloader obj>] when ds_item in
        # config has 1 item passed in a list
        if (
            self._validation_dl is not None
            and isinstance(self._validation_dl, (list, tuple))
            and len(self._validation_dl) > 1
        ):
            for _ in range(len(self._validation_dl)):
                self._validation_step_acc.append([])

        return self._validation_step_acc

    @validation_step_acc.setter
    def validation_step_acc(self, value):
        self._validation_step_acc = value


