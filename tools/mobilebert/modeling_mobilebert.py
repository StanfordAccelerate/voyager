# MIT License
#
# Copyright (c) 2020  The Google AI Language Team Authors, The HuggingFace Inc. team and github/lonePatient
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import MobileBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"
_CONFIG_FOR_DOC = "MobileBertConfig"
_TOKENIZER_FOR_DOC = "MobileBertTokenizer"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "mrm8488/mobilebert-finetuned-ner"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']"
_TOKEN_CLASS_EXPECTED_LOSS = 0.03

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "csarron/mobilebert-uncased-squad-v2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 3.98
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 13

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "lordtt13/emo-mobilebert"
_SEQ_CLASS_EXPECTED_OUTPUT = "'others'"
_SEQ_CLASS_EXPECTED_LOSS = "4.72"

MOBILEBERT_PRETRAINED_NETWORK_ARCHIVE_LIST = ["google/mobilebert-uncased"]


class NoNorm(nn.Module):
    def __init__(self, feat_size, eps=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


NORM2FN = {"layer_norm": nn.LayerNorm, "no_norm": NoNorm}


class MobileBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = nn.Linear(embedded_input_size, config.hidden_size)

        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = torch.cat(
                [
                    nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
                    inputs_embeds,
                    nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
                ],
                dim=2,
            )
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MobileBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = (context_layer,)
        return outputs


class MobileBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MobileBertSelfAttention(config)
        self.output = MobileBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        layer_input: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        attention_output = self.output(self_outputs[0], layer_input)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MobileBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class MobileBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)

    def forward(
        self, intermediate_states: torch.Tensor, residual_tensor_1: torch.Tensor, residual_tensor_2: torch.Tensor
    ) -> torch.Tensor:
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output


class BottleneckLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input


class Bottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)


class FFNOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class FFNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class MobileBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (
            (layer_output,)
            + outputs
            + (
                torch.tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs


class MobileBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # all_hidden_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        # )
        return hidden_states


class MobileBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.do_activate = config.classifier_activation
        if self.do_activate:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = torch.tanh(pooled_output)
            return pooled_output


class MobileBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MobileBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MobileBertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        hidden_states += self.decoder.bias
        return hidden_states


class MobileBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MobileBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output: torch.Tensor, pooled_output: torch.Tensor) -> Tuple[torch.Tensor]:
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_NETWORK_ARCHIVE_LIST
    # load_tf_weights = load_tf_weights_in_mobilebert
    base_model_prefix = "mobilebert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, NoNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MobileBertModel(MobileBertPreTrainedModel):
    """
    https://arxiv.org/pdf/2004.02984.pdf
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = MobileBertEmbeddings(config)
        self.encoder = MobileBertEncoder(config)

        self.pooler = MobileBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = None
            input_shape = input_ids.size()
            # raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        #     attention_mask, input_shape, self.device
        # )
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output = encoder_outputs[0]
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.mobilebert = MobileBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
