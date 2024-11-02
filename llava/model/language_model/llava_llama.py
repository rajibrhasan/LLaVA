#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..losses import *

local_rank = None

# use this function instead the standard print, to avoid verbose output in the logs
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_diff = DiffLoss()
        self.loss_sim = CMD()


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                embeds
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        rank0_print('==============================Text + Img Features1=========================')
        rank0_print('Input ids: ', input_ids)
        if position_ids is not None:
            rank0_print('Position_ids shape: ', position_ids.shape)
        rank0_print('Position_ids: ', position_ids)
        if attention_mask is not None:
            rank0_print('Attention mask shape: ', attention_mask.shape)
        rank0_print('Attention mask: ', attention_mask)
        if past_key_values is not None:
            rank0_print('Past key values shape: ', past_key_values.shape)
        rank0_print('Past key values: ', past_key_values)
        if labels is not None:
            rank0_print('Labels shape: ', labels.shape)
        rank0_print('Labels: ',labels)
        rank0_print('Input embeds shape: ', inputs_embeds.shape)

        rank0_print('==============================Img Features2=========================')
        rank0_print('Input ids: ', input_ids)
        if embeds['position_ids2'] is not None:
            rank0_print('Position ids shape: ', embeds['position_ids2'].shape)
        rank0_print('Position_ids: ', embeds['position_ids2'])
        if embeds['attention_mask2'] is not None:
            rank0_print('Attention mask shape: ', embeds['attention_mask2'].shape)
        rank0_print('Attention mask: ', embeds['attention_mask2'])
        if embeds['past_key_values'] is not None:
            rank0_print('Past key values shape: ', embeds['past_key_values'].shape)
        rank0_print('Past key values: ', embeds['past_key_values'])
        if embeds['new_labels2'] is not None:
            rank0_print('New labels 2 shape: ', embeds['new_labels2'].shape)
        rank0_print('Labels: ', embeds['new_labels2'])
        rank0_print('Input embeds shape: ', embeds['new_input_embeds2'].shape)

        outputs =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        device = outputs['loss'].device
        if embeds is not None:
            diff_loss = self.loss_diff(embeds['img_embeds1'], embeds['img_embeds2']) 
            # diff_loss += self.loss_diff(embeds['img_embeds2'], embeds['text_embeds'])
            # sim_loss = self.loss_sim(embeds['img_embeds2'], embeds['text_embeds'], 5)
            outputs['loss'] += self.config.diff_loss_coef * diff_loss.to(device)
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        kwargs.pop("cache_position", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
