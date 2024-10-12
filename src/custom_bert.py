"""PyTorch Custom BERT model."""

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F

from transformers.file_utils import cached_path
# from transformers.utils import cached_file

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):  # BERT 配置类，用于存储模型的配置信息
    def __init__(self,
                vocab_size_or_config_json_file,  # 词汇表大小或配置文件路径
                hidden_size=768,  # 隐藏层维度
                num_hidden_layers=12,  # Transformer 层数
                num_attention_heads=12,  # 注意力头的数量
                intermediate_size=3072,  # 中间层的维度（前馈网络）
                hidden_act="gelu",  # 激活函数类型
                hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率
                attention_probs_dropout_prob=0.1,  # 注意力的 dropout 概率
                max_position_embeddings=512,  # 最大位置嵌入数
                type_vocab_size=2,  # 类型词汇表大小（区分不同句子）
                initializer_range=0.02):  # 初始化权重的范围

        # 如果输入是配置文件路径，加载 JSON 文件的内容
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value  # 动态设置配置项
        # 如果输入是词汇表大小，使用传入的参数初始化配置
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            # 输入参数不合法时抛出异常
            raise ValueError("第一个参数必须是词汇表大小 (int) 或预训练模型的配置文件路径 (str)")

    # 从字典中构建 `BertConfig` 实例
    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    # 从 JSON 文件中构建 `BertConfig` 实例
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    # 返回对象的字符串表示形式
    def __repr__(self):
        return str(self.to_json_string())

    # 将配置序列化为字典
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    # 将配置序列化为 JSON 字符串
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertLayerNorm(nn.Module):  # 定义 BERT 层归一化
    def __init__(self, hidden_size, eps=1e-5):
        """创建层归一化模块，类似于 TensorFlow 中的实现。"""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重为 1
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # 初始化偏置为 0
        self.variance_epsilon = eps  # 防止分母为 0 的小常数

    def forward(self, x):
        # 计算均值 u
        u = x.mean(-1, keepdim=True)
        # 计算方差 s
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # 归一化处理
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias  # 应用权重和偏置

class BertEmbeddings(nn.Module):  # 定义 BERT 嵌入层
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # 词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # 位置嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 句子类型嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 层归一化和 dropout
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)  # 获取序列长度
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # 生成位置索引
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # 如果未指定 token_type_ids，全部初始化为 0

        # 计算词嵌入、位置嵌入和句子类型嵌入
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将三者相加并进行归一化和 dropout
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # 返回最终的嵌入结果

class BertSelfAttention(nn.Module):  # 定义 BERT 自注意力机制
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # 确保隐藏层大小是注意力头数的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "隐藏层大小 (%d) 不能被注意力头数 (%d) 整除" % (config.hidden_size, config.num_attention_heads))
        
        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义线性变换层，用于生成 query, key, value
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout，用于防止过拟合
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 用于将张量变换为注意力得分计算的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # 计算 query, key, value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 转换为多头注意力的形状
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 添加注意力掩码
        attention_scores = attention_scores + attention_mask

        # 计算注意力分布
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文层输出
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs  # 返回上下文层和注意力分布



class BertSelfOutput(nn.Module):  # 定义自注意力输出层，继承自 nn.Module
    def __init__(self, config):
        # 初始化自注意力输出层
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 全连接层，用于线性变换
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)  # 层归一化，用于稳定训练
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 随机丢弃部分神经元，防止过拟合

    def forward(self, hidden_states, input_tensor):
        """
        前向传播函数，处理自注意力的输出。

        参数：
        - hidden_states：来自自注意力模块的输出
        - input_tensor：输入张量，通常是来自前一层的输出

        返回：
        - 经过线性变换、归一化和残差连接的最终输出
        """
        
        hidden_states = self.dense(hidden_states)  # 对自注意力输出进行线性变换
        hidden_states = self.dropout(hidden_states)  # 进行 dropout 操作，防止过拟合
        # 残差连接，将原输入 `input_tensor` 加入线性变换后的结果，并进行层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states  # 返回处理后的张量

class BertAttention(nn.Module):  # 定义自注意力机制，继承自 nn.Module
    def __init__(self, config):
        # 初始化自注意力模块
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)  # 初始化自注意力（self-attention）模块
        self.output = BertSelfOutput(config)  # 初始化自注意力输出模块

    def forward(self, input_tensor, attention_mask):
        """
        前向传播函数，执行自注意力机制的计算。

        参数：
        - input_tensor：输入张量，通常是前一层的输出
        - attention_mask：注意力掩码，控制哪些 token 需要被关注

        返回：
        - attention_output：经过自注意力机制后的输出
        - att_score：自注意力分数，用于衡量各 token 之间的相关性
        """
        
        # 通过自注意力模块计算自注意力输出和注意力分数
        self_output, att_score = self.self(input_tensor, attention_mask)
        # 将自注意力的输出通过 BertSelfOutput 进行进一步处理，包括残差连接和层归一化
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_score  # 返回最终的注意力输出和注意力分数



class BertIntermediate(nn.Module):  # 定义 BERT 模型中的前馈神经网络的中间层
    def __init__(self, config):
        # 初始化中间层
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)  # 线性变换，将 hidden_size 转为 intermediate_size
        # 激活函数，根据配置选择，如果是字符串则从 ACT2FN 字典中查找，否则直接使用传入的函数
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        """
        前向传播函数，执行中间层的计算。

        参数：
        - hidden_states：输入的隐藏状态（来自上一层）
        - tgt_pos：目标位置，通常是 [MASK] 的位置
        - tmp_score：临时得分，用于调整目标位置的隐藏状态
        - imp_pos：重要位置，指定需要修改的重要性位置
        - imp_op：重要性操作，控制如何处理重要性

        返回：
        - 返回经过激活函数处理的隐藏状态，若 `imp_op == 'return'`，则返回重要性权重
        """

        # 对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)  # [batch_size, max_len, intermediate_size]

        # 如果提供了临时得分，则将目标位置的隐藏状态替换为临时得分，得到了每一步的隐藏状态
        if tmp_score is not None:
            hidden_states[:, tgt_pos, :] = tmp_score

        # 如果需要返回重要性权重，则初始化列表
        if imp_op == 'return':
            imp_weights = []

        # 如果指定了重要性位置，则根据操作类型进行处理
        if imp_pos is not None:
            for layer, pos in imp_pos:
                if imp_op == 'remove':  # 如果操作是 'remove'，将目标位置的某些维度设置为 0
                    hidden_states[:, tgt_pos, pos] = 0.0
                if imp_op == 'enhance':  # 如果操作是 'enhance'，将目标位置的某些维度放大两倍
                    hidden_states[:, tgt_pos, pos] *= 2.0
                if imp_op == 'return':  # 如果操作是 'return'，记录该位置的权重
                    imp_weights.append(hidden_states[0, tgt_pos, pos].item())

        # 如果 imp_op 是 'return'，则返回隐藏状态和重要性权重
        if imp_op == 'return':
            return hidden_states, imp_weights
        else:
            # 否则仅返回隐藏状态
            return hidden_states

class BertOutput(nn.Module):  # 定义 BERT 模型中的输出层
    def __init__(self, config):
        # 初始化输出层
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)  # 线性层，将 intermediate_size 映射回 hidden_size
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)  # 层归一化，用于稳定训练
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 随机丢弃部分神经元，防止过拟合

    def forward(self, hidden_states, input_tensor):
        """
        前向传播函数，处理前馈神经网络的输出。

        参数：
        - hidden_states：来自前馈网络中间层的隐藏状态
        - input_tensor：来自前一层的原始输入

        返回：
        - 返回经过线性变换、归一化和残差连接的最终输出
        """

        # 通过线性层将中间层输出映射回 hidden_size 维度
        hidden_states = self.dense(hidden_states)
        # 进行 dropout 操作，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 残差连接，将原输入 input_tensor 加入线性变换后的结果，并进行层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states  # 返回最终输出



class BertLayer(nn.Module):  # 定义 BERT 模型中的一个 Transformer 层，继承自 nn.Module
    def __init__(self, config):
        # 初始化 BERT 层，传入配置参数 config
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)  # 定义自注意力机制模块
        self.intermediate = BertIntermediate(config)  # 定义前馈神经网络中的中间层
        self.output = BertOutput(config)  # 定义输出层，结合自注意力和前馈神经网络的输出

    def forward(self, hidden_states, attention_mask, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        """
        前向传播函数，处理输入并逐步通过自注意力、前馈网络和输出层。

        参数说明：
        - hidden_states：输入的隐藏状态
        - attention_mask：注意力掩码，用于控制自注意力机制的计算
        - tgt_pos：目标位置（可选），用于指定 [MASK] 的位置
        - tmp_score：临时得分（可选），用于计算梯度或某些特定任务
        - imp_pos：重要位置（可选），用于指定计算重要性的位置
        - imp_op：重要性操作，决定是否返回重要性权重

        返回值：
        - 返回层的最终输出和中间层输出。如果 imp_op 为 'return'，还返回重要性权重。
        """
        
        # 通过自注意力模块处理输入，得到注意力输出和注意力分数
        attention_output, att_score = self.attention(hidden_states, attention_mask)

        # 判断是否需要返回重要性权重
        if imp_op == 'return':
            # 通过前馈网络中的中间层处理注意力输出，并计算重要性权重
            intermediate_output, imp_weights = self.intermediate(
                attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
        else:
            # 通过前馈网络中的中间层处理注意力输出（不计算重要性权重）
            intermediate_output = self.intermediate(
                attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)

        # 通过输出层整合中间层的输出和注意力输出，得到最终层的输出
        layer_output = self.output(intermediate_output, attention_output)

        # 如果需要返回重要性权重，则返回最终层输出、中间层输出和重要性权重
        if imp_op == 'return':
            return layer_output, intermediate_output, imp_weights
        else:
            # 否则只返回最终层输出和中间层输出
            return layer_output, intermediate_output



class BertEncoder(nn.Module):  # 定义 BERT 编码器类，继承自 PyTorch 的 nn.Module
    def __init__(self, config):
        # 初始化 BERT 编码器，传入配置参数 config
        super(BertEncoder, self).__init__()
        # 创建多个 BertLayer，每个层代表 BERT 的一个 Transformer 层
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, tgt_layer=None, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        """
        前向传播函数，逐层对输入进行处理。

        参数说明：
        - hidden_states：上一层的隐藏状态，作为当前层的输入
        - attention_mask：注意力掩码，用于控制自注意力机制的计算
        - tgt_layer：目标层，用于特定任务（如重要性计算）
        - tgt_pos：目标位置，用于指定 [MASK] 的位置
        - tmp_score：临时得分，用于计算梯度或特定任务
        - imp_pos：重要位置，通常用于重要性评分任务
        - imp_op：重要性操作，决定是否返回重要性权重

        返回值：
        - all_encoder_layers：每一层的输出
        - ffn_weights：前馈神经网络（FFN）层的权重
        - imp_weights：重要性权重（如果 imp_op 为 'return'）
        """

        all_encoder_layers = []  # 用于存储每一层的输出
        ffn_weights = None  # 初始化 FFN 层的权重
        if imp_op == 'return':
            imp_weights = []  # 如果需要返回重要性权重，初始化该列表

        # 遍历每一层 Transformer
        for layer_index, layer_module in enumerate(self.layer):
            # 如果传入了重要位置，则筛选出当前层的 imp_pos
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
            else:
                imp_pos_at_this_layer = None

            # 如果需要返回重要性权重
            if imp_op == 'return':
                if tgt_layer == layer_index:  # 如果当前层是目标层
                    hidden_states, ffn_weights, imp_weights_l = layer_module(
                        hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, 
                        imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:  # 非目标层，不计算 tmp_score，只返回 imp_weights
                    hidden_states, _, imp_weights_l = layer_module(
                        hidden_states, attention_mask, tgt_pos=tgt_pos, 
                        imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                imp_weights.extend(imp_weights_l)  # 将该层的 imp_weights 加入列表
            else:
                # 不需要重要性权重的情况
                if tgt_layer == layer_index:  # 如果是目标层，则计算 ffn_weights 和 hidden_states
                    hidden_states, ffn_weights = layer_module(
                        hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, 
                        imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:  # 非目标层，继续传播 hidden_states
                    hidden_states, _ = layer_module(
                        hidden_states, attention_mask, tgt_pos=tgt_pos, 
                        imp_pos=imp_pos_at_this_layer, imp_op=imp_op)

        # 保存最后一层的输出
        all_encoder_layers.append(hidden_states)

        # 如果 imp_op 为 'return'，返回所有层的输出、FFN 层权重和重要性权重
        if imp_op == 'return':
            return all_encoder_layers, ffn_weights, imp_weights
        else:
            # 否则只返回所有层的输出和 FFN 层权重
            return all_encoder_layers, ffn_weights



class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                bert_model_embedding_weights.size(0),
                                bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class PreTrainedBertModel(nn.Module):

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):  # 定义 BERT 模型，继承自预训练的 BERT 模型基类

    def __init__(self, config):
        # 初始化 BERT 模型，传入配置参数 config
        super(BertModel, self).__init__(config)  # 调用父类的初始化方法
        self.embeddings = BertEmbeddings(config)  # 初始化 BERT 的词嵌入层
        self.encoder = BertEncoder(config)  # 初始化 BERT 的编码器（多层 Transformer）
        self.apply(self.init_bert_weights)  # 应用初始化权重的函数

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, tgt_pos=None, tgt_layer=None, tmp_score=None, imp_pos=None, imp_op=None):
        """
        前向传播函数，计算 BERT 的输出。

        参数说明：
        - input_ids：输入的 token ID 序列
        - attention_mask：注意力掩码，用于标识需要注意的 token（可选）
        - token_type_ids：表示句子类型的标识（可选）
        - tgt_pos：目标位置，用于标识需要预测的 [MASK] 位置
        - tgt_layer：目标层，指定 BERT 模型中的某一层用于特定任务
        - tmp_score：临时得分，用于调试或某些特定的计算
        - imp_pos：重要位置，用于特定任务，比如重要性评分的位置
        - imp_op：重要性操作，用于控制模型是否返回重要性权重

        返回值：
        - 返回最后一层的输出、FFN 层权重，如果指定了重要性操作，还会返回重要性权重
        """
        
        # 如果没有提供 attention_mask（注意力掩码），则创建一个全 1 的掩码，表示所有位置都要注意
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 如果没有提供 token_type_ids（区分句子类型的标识），则创建全 0 的标识
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 扩展 attention_mask 的维度，以便与后续的计算匹配
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 将注意力掩码的类型转换为与模型参数一致，确保与 fp16（半精度浮点数）兼容
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # 转换掩码的值：将1变为0（正常位置），将0变为一个非常小的负数（屏蔽位置），以排除不需要注意的位置
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 计算嵌入层的输出，输入为 input_ids 和 token_type_ids
        embedding_output = self.embeddings(input_ids, token_type_ids)  # [num_steps, seq_len, hidden_size]
 
        # 判断是否需要返回重要性权重
        if imp_op == 'return':
            # 编码器返回每层的输出、FFN 层的权重和重要性权重
            encoded_layers, ffn_weights, imp_weights = self.encoder(
                embedding_output, 
                extended_attention_mask,
                tgt_layer=tgt_layer, 
                tgt_pos=tgt_pos, 
                tmp_score=tmp_score, 
                imp_pos=imp_pos, 
                imp_op=imp_op
            )
        else:
            # 如果不需要返回重要性权重，则只返回每层的输出和 FFN 层的权重
            encoded_layers, ffn_weights = self.encoder(
                embedding_output, 
                extended_attention_mask,
                tgt_layer=tgt_layer, 
                tgt_pos=tgt_pos, 
                tmp_score=tmp_score, 
                imp_pos=imp_pos, 
                imp_op=imp_op
            )

        # 获取最后一层的编码结果
        sequence_output = encoded_layers[-1]

        # 如果需要返回重要性权重，则返回最后一层的输出、FFN 层权重和重要性权重
        if imp_op == 'return':
            return sequence_output, ffn_weights, imp_weights
        else:
            # 否则只返回最后一层的输出和 FFN 层权重
            return sequence_output, ffn_weights



class BertForMaskedLM(PreTrainedBertModel):  # 定义一个用于掩码语言模型的BERT类，继承自预训练的BERT模型类

    def __init__(self, config):
        # 初始化BERT模型，传入模型配置 config
        super(BertForMaskedLM, self).__init__(config)  # 调用父类的初始化方法
        self.bert = BertModel(config)  # 使用配置初始化BERT模型
        # 定义用于掩码语言模型任务的头部，它使用 BERT 嵌入层的词嵌入权重
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        # 初始化 BERT 模型的权重
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None, imp_pos=None, imp_op=None):
        """
        前向传播函数，计算模型的输出。

        参数说明：
        - input_ids：输入的 token id 序列
        - token_type_ids：表示句子类型的标识（可选）
        - attention_mask：注意力掩码，用于标识需要注意的 token（可选）
        - tgt_pos：目标位置，用于标识需要预测的 [MASK] 位置
        - tgt_layer：目标层，指定BERT模型中的哪一层用于特定任务
        - tmp_score：临时得分，用于调试或某些特定的计算（如重要性评分）
        - tgt_label：目标标签，表示用于计算损失或梯度的标签
        - imp_pos：重要位置，用于特定任务，比如重要性评分的位置
        - imp_op：重要性操作，用于控制模型是否需要返回重要性权重

        返回值：
        - 可能返回不同类型的输出，视 imp_op 和 tmp_score 的设置而定
        """
        
        # 如果 tmp_score 不为空，则意味着我们需要扩展输入 batch 的维度
        if tmp_score is not None:
            # 获取 tmp_score 的 batch 大小
            batch_size = tmp_score.shape[0]
            # 将 input_ids、token_type_ids、attention_mask 重复 batch_size 次
            input_ids = input_ids.repeat(batch_size, 1)  # [num_steps, seq_len]
            token_type_ids = token_type_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)

        # 判断是否需要返回重要性权重
        if imp_op == 'return':
            # 当 imp_op 为 'return' 时，模型不仅返回最后一层的隐藏状态，还返回 FFN 层和重要性权重
            last_hidden, ffn_weights, imp_weights = self.bert(
                input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, 
                tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op
            )  # 返回 (batch, max_len, hidden_size), (batch, max_len, ffn_size), (n_imp_pos)
        else:
            # 否则，仅返回最后一层隐藏状态和 FFN 层权重
            last_hidden, ffn_weights = self.bert(
                input_ids=input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, 
                tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op
            )  # 返回 (batch, max_len, hidden_size), (batch, max_len, ffn_size)

        # 从最后一层隐藏状态中提取目标位置的表示，即对应 [MASK] 的隐藏状态
        last_hidden = last_hidden[:, tgt_pos, :]  # (batch, hidden_size)
        # 从 FFN 层权重中提取目标位置的表示
        ffn_weights = ffn_weights[:, tgt_pos, :]  # (batch, ffn_size)

        # 使用分类头部计算目标位置的分类 logits
        tgt_logits = self.cls(last_hidden)  # (batch, n_vocab)
        # 通过 softmax 计算分类的概率分布
        tgt_prob = F.softmax(tgt_logits, dim=1)  # (batch, n_vocab)

        # 如果 imp_op 为 'return'，仅返回重要性权重
        if imp_op == 'return':
            return imp_weights
        else:
            # 如果没有传入 tmp_score，则返回 FFN 层权重和分类 logits
            if tmp_score is None:
                return ffn_weights, tgt_logits
            else:
                # 否则，计算分类概率的梯度，返回分类概率和梯度值
                gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, tgt_label]), tmp_score)
                return tgt_prob, gradient[0]
