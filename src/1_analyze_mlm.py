"""
BERT MLM runner
"""
from typing import *
import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

import transformers
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """
    将一个输入样例转换为 BERT 的输入特征。

    参数：
    - example：输入样例，包含文本、标签和关系信息
    - max_seq_length：最大序列长度，超过该长度的序列将被截断
    - tokenizer：用于将文本标记化并转换为 token id 的 BERT 分词器

    返回：
    - features：用于 BERT 输入的特征，包括 input_ids、input_mask、segment_ids 和 baseline_ids
    - tokens_info：包含处理后的 token 信息和目标对象的标签
    """
    
    features = []  # 用于存储 BERT 输入特征
    tokenslist = []  # 用于存储 token 序列（暂未使用）

    # 对样例文本进行分词，将文本转换为 token 列表
    ori_tokens = tokenizer.tokenize(example[0])
    # 模板都很简单，几乎不会超过最大长度限制。如果超过，截断到 max_seq_length - 2（[CLS] 和 [SEP] 占用两个位置）
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # 添加特殊 token [CLS] 和 [SEP]，用于标记句子的起始和结束
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    # 基准 tokens 全部使用 [UNK] 作为占位符
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    # 生成 segment_ids（句子类型标记），全为 0，因为这里只有一个句子
    segment_ids = [0] * len(tokens)

    # 将 token 序列转换为 token id
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    # attention mask，用于标记哪些 token 需要被注意，实际 token 为 1，填充部分为 0
    input_mask = [1] * len(input_ids)

    # 如果 token 不够 max_seq_length 长度，使用 [PAD] token 进行填充（在 BERT 中 [PAD] token 的 id 是 0）
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding  # 填充 input_ids
    baseline_ids += padding  # 填充 baseline_ids
    segment_ids += padding  # 填充 segment_ids
    input_mask += padding  # 填充 attention mask

    # 确保所有序列都被正确填充到 max_seq_length 长度
    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # 构建用于 BERT 输入的特征字典
    features = {
        'input_ids': input_ids,  # 输入的 token id 序列
        'input_mask': input_mask,  # attention mask
        'segment_ids': segment_ids,  # 句子类型标记
        'baseline_ids': baseline_ids,  # 基准 token id 序列（用于集成梯度等任务）
    }
    
    # 生成 tokens 的信息，包括 token 序列、关系、真实目标和预测目标
    tokens_info = {
        "tokens": tokens,  # token 列表
        "relation": example[2],  # 样例中的关系
        "gold_obj": example[1],  # 样例中的真实目标（标签）
        "pred_obj": None  # 预测目标（暂未预测）
    }

    return features, tokens_info  # 返回生成的特征和 tokens 信息



def scaled_input(emb, batch_size, num_batch):
    """
    生成 scaled 输入，用于集成梯度计算。
    
    参数:
    - emb: 输入的嵌入表示，形状为 (1, ffn_size)。
    - batch_size: 批次大小。
    - num_batch: 批次数量。
    
    返回:
    - res: 逐步变化的嵌入表示，形状为 (num_points, ffn_size)。
    - step: 每一步的变化步长，形状为 (ffn_size)。
    """
    baseline = torch.zeros_like(emb)  # 创建全零的基准嵌入，形状与 emb 相同 (1, ffn_size)

    num_points = batch_size * num_batch  # 总的变化步数
    # 计算每一步变化的步长
    step = (emb - baseline) / num_points  # (1, ffn_size)

    # 逐步生成从 baseline 到 emb 的插值，生成 num_points 个不同的嵌入表示
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]  # 返回生成的嵌入表示和步长


def convert_to_triplet_ig(ig_list):
    """
    将集成梯度值转换为三元组格式，只保留大于最大值 10% 的值。
    
    参数:
    - ig_list: 集成梯度列表，形状为 (num_layers, ffn_size)。
    
    返回:
    - ig_triplet: 集成梯度的三元组格式列表，保留显著贡献的值。
    """
    ig_triplet = []  # 用于存储集成梯度三元组
    ig = np.array(ig_list)  # 将集成梯度列表转换为 numpy 数组，形状为 (num_layers, ffn_size)
    max_ig = ig.max()  # 找到集成梯度的最大值

    # 遍历每个层和每个位置的集成梯度值
    for i in range(ig.shape[0]):  # 遍历每一层
        for j in range(ig.shape[1]):  # 遍历每个前馈网络维度
            # 如果集成梯度值大于最大值的 10%，则保留该值
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])  # 记录 (层数, 位置, 集成梯度值)

    return ig_triplet  # 返回三元组格式的集成梯度列表



def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument("--data_path",  # 输入数据的路径，必须是 .json 文件
                        default=None,
                        type=str,
                        required=True,
                        help="输入数据路径，必须是用于MLM任务的 .json 文件。")
    parser.add_argument("--tmp_data_path",  # 临时输入数据路径，可选参数
                        default=None,
                        type=str,
                        help="临时输入数据路径，必须是用于MLM任务的 .json 文件。")
    parser.add_argument("--bert_model",  # 指定使用的预训练BERT模型，如 bert-base-uncased 等
                        default=None, 
                        type=str, 
                        required=True, 
                        help="选择BERT预训练模型，如 bert-base-uncased、bert-large-uncased 等。")
    parser.add_argument("--output_dir",  # 输出目录，保存模型预测结果和检查点
                        default=None,
                        type=str,
                        required=True,
                        help="输出目录，保存模型预测和检查点。")
    parser.add_argument("--output_prefix",  # 输出文件前缀，用于标识每次实验的结果
                        default=None,
                        type=str,
                        required=True,
                        help="输出前缀，用于标识每次实验的运行。")

    # 其他参数
    parser.add_argument("--max_seq_length",  # 最大输入序列长度，超过此长度会被截断，短于此长度会被填充
                        default=128,
                        type=int,
                        help="WordPiece分词后输入序列的最大长度，超出此长度的序列将被截断，短于此长度的序列将被填充。")
    parser.add_argument("--do_lower_case",  # 是否将输入转为小写，适用于 uncased 模型
                        default=False,
                        action='store_true',
                        help="如果使用 uncased 模型，设置此标志将输入转为小写。")
    parser.add_argument("--no_cuda",  # 是否禁用CUDA（GPU）
                        default=False,
                        action='store_true',
                        help="是否在可用时禁用CUDA（GPU）。")
    parser.add_argument("--gpus",  # 可用的GPU ID
                        type=str,
                        default='0',
                        help="可用的GPU ID。")
    parser.add_argument('--seed',  # 随机种子，用于结果的可重复性
                        type=int,
                        default=42,
                        help="初始化的随机种子。")
    parser.add_argument("--debug",  # 调试时使用的样本数量，-1 表示不调试
                        type=int,
                        default=-1,
                        help="调试样本的数量，-1 表示不进行调试。")
    parser.add_argument("--pt_relation",  # 要计算关系簇的特定关系，选填
                        type=str,
                        default=None,
                        help="用于计算关系簇的关系。")

    # 集成梯度相关参数
    parser.add_argument("--get_pred",  # 是否获取预测结果
                        action='store_true',
                        help="是否获取预测结果。")
    parser.add_argument("--get_ig_pred",  # 是否计算预测标签的集成梯度
                        action='store_true',
                        help="是否获取预测标签的集成梯度。")
    parser.add_argument("--get_ig_gold",  # 是否计算真实标签的集成梯度
                        action='store_true',
                        help="是否获取真实标签的集成梯度。")
    parser.add_argument("--get_base",  # 是否获取基础值（FFN权重）
                        action='store_true',
                        help="是否获取基础值。")
    parser.add_argument("--batch_size",  # 批处理大小
                        default=16,
                        type=int,
                        help="切分时的总批处理大小。")
    parser.add_argument("--num_batch",  # 每个样本的批处理数量
                        default=10,
                        type=int,
                        help="每个样本的批处理数量。")

    # 解析命令行参数
    args = parser.parse_args()

    # 设置设备（CPU或GPU）
    if args.no_cuda or not torch.cuda.is_available():  # 如果禁用CUDA或CUDA不可用，使用CPU
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:  # 如果指定了一个GPU，使用该GPU
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # TODO: 实现多GPU并行计算
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 保存命令行参数到文件中
    os.makedirs(args.output_dir, exist_ok=True)  # 如果输出目录不存在，则创建
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # 加载预训练的BERT模型
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()  # 清除CUDA缓存
    model = BertForMaskedLM.from_pretrained(args.bert_model)  # 加载用于掩码语言模型的预训练BERT模型
    model.to(device)  # 将模型转移到指定设备（CPU或GPU）

    # 数据并行处理（如果有多个GPU）
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()  # 将模型设置为评估模式

    # 准备评估数据集
    if os.path.exists(args.tmp_data_path):  # 如果临时数据路径存在，直接加载
        with open(args.tmp_data_path, 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else:  # 否则，从指定数据路径加载并处理
        with open(args.data_path, 'r') as f:
            eval_bag_list_all:Dict[str,List[List[List[str]]]] = json.load(f)  # 键是关系类别，值是三维列表，其中每个元素是一个batch，每个样本中有3个元素，分别是句子、答案、问题类别
        # 将评估数据按关系分组
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]  # 提取关系名
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        # 保存处理好的数据
        with open(args.tmp_data_path, 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)

    # 评估每个关系下的batch包
    for relation, eval_bag_list in eval_bag_list_perrel.items():
        if args.pt_relation is not None and relation != args.pt_relation:
            continue
        # 记录运行时间
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '-' + relation + '.rlt' + '.jsonl'), 'w') as fw:
            for bag_idx, eval_bag in enumerate(eval_bag_list):  # 遍历每个batch
                res_dict_bag = []  # 用于存储一个batch的结果
                for eval_example in eval_bag:  # 遍历batch中的每个样本
                    # 将评估样本转换为特征
                    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                    # 将特征转换为长整型Tensor
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    # 升维后[batch_size=1, seq_len]
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    baseline_ids = baseline_ids.to(device)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)

                    # 记录真实输入长度（也包括[CLS]和[SEP]）
                    input_len = int(input_mask[0].sum())

                    # 记录 [MASK] 位置，即待预测位置
                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    # 用于存储一个样本的结果
                    res_dict = {
                        'pred': [],  # 预测结果
                        'ig_pred': [],  # 预测标签的集成梯度
                        'ig_gold': [],  # 真实标签的集成梯度
                        'base': []  # 基础FFN权重
                    }

                    # 获取原始预测概率
                    if args.get_pred:
                        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
                        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
                        res_dict['pred'].append(base_pred_prob.tolist())

                    for tgt_layer in range(model.bert.config.num_hidden_layers):  # 遍历BERT架构中的每一个encoder层
                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                        pred_label = int(torch.argmax(logits[0, :]))  # scalar，预测的label在词表中的ID
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                        scaled_weights.requires_grad_(True)

                        # integrated grad at the pred label for each layer
                        if args.get_ig_pred:
                            ig_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_pred = grad if ig_pred is None else torch.add(ig_pred, grad)  # (ffn_size)
                            ig_pred = ig_pred * weights_step  # (ffn_size)
                            res_dict['ig_pred'].append(ig_pred.tolist())

                        # integrated grad at the gold label for each layer
                        if args.get_ig_gold:
                            ig_gold = None
                            for batch_idx in range(args.num_batch):  # 遍历每一个步数batch
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]  # [num_steps, intermediate_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
                            ig_gold = ig_gold * weights_step  # (ffn_size)
                            res_dict['ig_gold'].append(ig_gold.tolist())

                        # base ffn_weights for each layer
                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args.get_ig_gold:
                        res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
                    if args.get_base:
                        res_dict['base'] = convert_to_triplet_ig(res_dict['base'])

                    res_dict_bag.append([tokens_info, res_dict])

                fw.write(res_dict_bag)

        # record running time
        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()