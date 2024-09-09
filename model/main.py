import tensorflow as tf
# import tensorflow.contrib.eager as tfe
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import json
import data_utils

from data_utils import prepare_data, get_tree_vocab, get_seq_vocab, data2id_v2, get_type_vocab, post_processing

from dataset import Dataset_V2

import argparse
from tree2seq_with_attn_v2 import Tree2SeqModel
from evaluate_utils import bleu_fn, rouge_fn, bleu_rouge_fn, compare_with_best_results
from treelstm.tree import BinaryTree, NaryTree

from config import get_data_config

# from gpu_utils import GPUChecker
import math
# from metric_checkpoint_saver import MetricCheckpointSaver

def add_arguments(parser, data_config):

    parser.add_argument('--init_weight', type=float, default=0.1, help='')
    parser.add_argument('--embed_size', type=int, default=data_config.embed_size, help='')
    parser.add_argument('--hidden_size', type=int, default=data_config.hidden_size, help='')
    parser.add_argument('--decoder_num_layers', type=int, default=1, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--num_epoch', type=int, default=data_config.num_epoch, help='')
    parser.add_argument('--max_iterations', type=int, default=data_config.max_iterations, help='')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='')
    parser.add_argument('--eval_begin_step', type=int, default=data_config.eval_begin_step, help='evaluation is conducted must larger than this step number')
    parser.add_argument('--eval_every_steps', type=int, default=data_config.eval_every_steps, help='every this steps, evaluation is conducted')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--rand_seed', type=int, default=1234, help="set the fix random seed to randomize")
    parser.add_argument('--layer_norm', type=bool, default=False, help="add layer normalization to encoder decoder lstm")
    parser.add_argument('--l2_loss', type=bool, default=False, help='add l2 loss to the decoder last projection layer')
    parser.add_argument('--l2_weight', type=float, default=3e-5, help='l2 weight to multiply the l2 loss')
    parser.add_argument('--max_child_num', type=int, default=data_config.max_child_num, help='each tree max child num in train , dev, test set')
    parser.add_argument('--copy', type=bool, default=True, help='whether to copy from input')
    parser.add_argument('--copy_once', type=bool, default=False, help='if True, in each time step, just copy the node which has not been ever copied before')
    parser.add_argument('--predict_copy_once', type=bool, default=False, help='if True, different from copy_once, it will mask the copy probs in the training process')
    parser.add_argument('--copy_decay', type=bool, default=False, help='if True, the probs of previously copied word will be decay following some strategy')
    parser.add_argument('--copy_decay_keep_prob', type=float, default=data_config.copy_decay_keep_prob, help='if copy_decay is True, for the copy distance, current time step the quantity of decay is power(copy_decay_keep_prob, copy_distance)')
    parser.add_argument('--classifier', type=str, default='rl', help='in copy pattern, valid values are hard\soft\gumbel\\rl, if gumbel chosen, use gumbel softmax to sample the classifier value, not the hard classifier or soft classifier;if rl, use gumbel trick to sample both classifier and output distribution')
    parser.add_argument('--metric', type=str, default='bleu', help='evaluation metric, bleu/rouge/both')
    parser.add_argument('--reward', type=str, default='bleu', help='in rl mode, the reward type, bleu or rouge')
    parser.add_argument('--mixed_train_weight', type=float, default=0.3, help='when the classifier is rl and this argument is not None, it will train mixing the MLE and RL loss')
    parser.add_argument('--pretrain_steps', type=int, default=data_config.pretrain_steps, help='pretrain steps works, when classifier is rl')
    parser.add_argument('--min_vocab_count', type=int, default=data_config.min_vocab_count, help='vocab appearence count must larger or equal this min vocab count')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature, use in gumbel softmax')

    parser.add_argument('--train_file', type=str, default=data_config.train_file, help='')
    parser.add_argument('--dev_file', type=str, default=data_config.dev_file, help='')
    parser.add_argument('--test_file', type=str, default=data_config.test_file, help='')
    parser.add_argument('--tree_vocab_file', type=str, default=data_config.tree_vocab_file, help='')
    parser.add_argument('--seq_vocab_file', type=str, default=data_config.seq_vocab_file, help='')
    parser.add_argument('--use_type', type=bool, default=True, help='use type to choose training weights')
    parser.add_argument('--use_source_seq', type=bool, default=True, help='use tree sequence data to sequence encoder')
    parser.add_argument('--bidirectional', type=bool, default=False, help='if True , add bidirectional lstm to source seq encode (use_source_seq should be True)')
    parser.add_argument('--type_vocab_file', type=str, default=data_config.type_vocab_file, help='type vocab to create type embedding')
    parser.add_argument('--primitive_type', type=bool, default=False, help='a flag to indicate whether to use primitive type in copy machanism')
    parser.add_argument('--primitive_type_file', type=str, default=data_config.primitive_type_file, help='primitive type file when apply copy machanism')
    parser.add_argument('--top_k_type', type=int, default=None, help='top k type to create type embedding according to word frequency')
    parser.add_argument('--distributed', type=bool, default=False, help='distributed training')
    parser.add_argument('--checkpoint_dir', type=str, default=data_config.checkpoint_dir, help='checkpoint saved directory')
    parser.add_argument('--rl_restore_checkpoint_dir', type=str, default=data_config.rl_restore_checkpoint_dir, help='use to save and restore weights between main session and reward session')
    parser.add_argument('--load_model', type=bool, default=False, help="if True means load model")
    parser.add_argument('--eval_result_log', type=str, default=data_config.eval_result_log, help="result log path while in evaluation")
    parser.add_argument('--predict_file', type=str, default=data_config.predict_file, help='')
    parser.add_argument('--save_input_to_file', type=str, default=data_config.save_input_to_file, help='if not None, will save formated input to this file prefix path ')
    parser.add_argument('--save_output_to_file', type=str, default=data_config.save_output_to_file, help='if not None, will save evaluated output to this file prefix path')
    # parser.add_argument('--iter_loss_log', type=str, default=data_config.iter_loss_log, help='record the loss value each iteration')
    parser.add_argument('--iter_eval_log', type=str, default=data_config.iter_eval_log, help='record the eval metric each iteration')


def prepare_train_data(params):

    primitive_types = []

    if params.primitive_type:
        with open(params.primitive_type_file, 'r', encoding='utf-8') as f:
            for line in f:
                primitive_types.append(line.strip())

    params.primitive_types = primitive_types
    print("get primitive_types:", primitive_types)

    train_tree_seqs = None
    dev_tree_seqs = None

    if params.use_source_seq:
        train_trees, train_seqs, train_tree_seqs = prepare_data(params.train_file,
                                                                primitive_types=params.primitive_types,
                                                                tree_class=NaryTree,
                                                                use_type=params.use_type,
                                                                use_source_seq=params.use_source_seq,
                                                                use_copy=params.copy,
                                                                whatset="train",
                                                                save_input_to_file=params.save_input_to_file)

        dev_trees, dev_seqs, dev_tree_seqs = prepare_data(params.dev_file,
                                                          primitive_types=params.primitive_types,
                                                          tree_class=NaryTree,
                                                          use_type=params.use_type,
                                                          use_source_seq=params.use_source_seq,
                                                          use_copy=params.copy,
                                                          whatset="dev",
                                                          save_input_to_file=params.save_input_to_file)
    else:
        train_trees, train_seqs = prepare_data(params.train_file,
                                               primitive_types=params.primitive_types,
                                               tree_class=NaryTree,
                                               use_type=params.use_type,
                                               use_copy=params.copy,
                                               whatset="train",
                                               save_input_to_file=params.save_input_to_file)

        dev_trees, dev_seqs = prepare_data(params.dev_file,
                                           primitive_types=params.primitive_types,
                                           tree_class=NaryTree,
                                           use_type=params.use_type,
                                           use_copy=params.copy,
                                           whatset="dev",
                                           save_input_to_file=params.save_input_to_file)

    t_word2idx = {}
    s_word2idx = {}

    get_tree_vocab(t_word2idx, train_trees, tree_seqs=train_tree_seqs, min_vocab_count=params.min_vocab_count)
    get_seq_vocab(s_word2idx, train_seqs, min_vocab_count=params.min_vocab_count)

    t_idx2word = {}
    s_idx2word = {}

    for word, idx in t_word2idx.items():
        t_idx2word[idx] = word

    for word, idx in s_word2idx.items():
        s_idx2word[idx] = word

    params.src_vocab_size = len(t_word2idx)
    params.tgt_vocab_size = len(s_word2idx)
    params.tree_word2idx = t_word2idx
    params.seq_word2idx = s_word2idx
    params.t_idx2word = t_idx2word
    params.s_idx2word = s_idx2word

    with open(params.tree_vocab_file, "w", encoding="utf-8") as tree_vocab_f:

        for w in t_word2idx:
            tree_vocab_f.write(w + "\t" + str(t_word2idx[w]) + "\n")

    with open(params.seq_vocab_file, "w", encoding="utf-8") as seq_vocab_f:

        for w in s_word2idx:
            seq_vocab_f.write(w + "\t" + str(s_word2idx[w]) + "\n")

    if params.use_type:
        type2idx = {}

        get_type_vocab(type2idx, train_trees, limit=params.top_k_type)

        params.type_vocab_size = len(type2idx)
        params.type2idx = type2idx
        print("type vocab size:", len(type2idx))
        print("type vocab:", type2idx)
        with open(params.type_vocab_file, "w", encoding="utf-8") as type_vocab_f:

            for t in type2idx:
                type_vocab_f.write(t + "\t" + str(type2idx[t]) + "\n")

    primitive_type_ids = []

    for pt in primitive_types:
        if pt in params.type2idx:
            primitive_type_ids.append(params.type2idx[pt])
        else:
            print("cannot find the primitive type %s in the type vocabulary !"%pt)

    params.primitive_type_ids = primitive_type_ids

    print("get primitive_type_ids:", primitive_type_ids)

    if params.use_type:
        train_dataset = data2id_v2(train_trees, train_seqs, t_word2idx, s_word2idx, type2idx=type2idx, tree_seqs=train_tree_seqs, copy=params.copy, whatset="train")
    else:
        train_dataset = data2id_v2(train_trees, train_seqs, t_word2idx, s_word2idx, tree_seqs=train_tree_seqs, copy=params.copy, whatset="train")

    if params.use_type:
        dev_res = data2id_v2(dev_trees, dev_seqs, t_word2idx, s_word2idx, type2idx=type2idx, tree_seqs=dev_tree_seqs, copy=params.copy, whatset="dev")
    else:
        dev_res = data2id_v2(dev_trees, dev_seqs, t_word2idx, s_word2idx, tree_seqs=dev_tree_seqs, copy=params.copy, whatset="dev")

    if params.copy:
        dev_dataset = dev_res[0]
        params.s_with_tree_in_seq_unk_word2idx = dev_res[1]
        params.s_with_tree_in_seq_unk_vocab_size = len(dev_res[1])

        s_with_tree_in_seq_unk_idx2word = {}

        for word, idx in params.s_with_tree_in_seq_unk_word2idx.items():
            s_with_tree_in_seq_unk_idx2word[idx] = word

        params.s_with_tree_in_seq_unk_idx2word = s_with_tree_in_seq_unk_idx2word

        print("add %d words unknown tree word in s_word2idx (devset): " % (len(params.s_with_tree_in_seq_unk_word2idx) - len(params.seq_word2idx)))
        print("seq_with_tree_in_seq_unk_vocab_size:", params.s_with_tree_in_seq_unk_vocab_size)
    else:
        dev_dataset = dev_res

    return train_dataset, dev_dataset


def prepare_test_data(params):

    t_word2idx = {}
    s_word2idx = {}

    with open(params.tree_vocab_file, "r", encoding="utf-8") as tree_vocab_f:

        for line in tree_vocab_f:
            split = line.strip().split()
            t_word2idx[split[0]] = int(split[1])

    with open(params.seq_vocab_file, "r", encoding="utf-8") as seq_vocab_f:

        for line in seq_vocab_f:
            split = line.strip().split()
            s_word2idx[split[0]] = int(split[1])

    t_idx2word = {}
    s_idx2word = {}

    for word, idx in t_word2idx.items():
        t_idx2word[idx] = word

    for word, idx in s_word2idx.items():
        s_idx2word[idx] = word

    params.src_vocab_size = len(t_word2idx)
    params.tgt_vocab_size = len(s_word2idx)
    params.tree_word2idx = t_word2idx
    params.seq_word2idx = s_word2idx
    params.t_idx2word = t_idx2word
    params.s_idx2word = s_idx2word

    if params.use_type:

        type2idx = {}
        with open(params.type_vocab_file, "r", encoding="utf-8") as type_vocab_f:

            for line in type_vocab_f:
                split = line.strip().split()
                type2idx[split[0]] = int(split[1])

        params.type_vocab_size = len(type2idx)
        params.type2idx = type2idx
        print("type vocab size:", len(type2idx))
        print("type vocab:", type2idx)

    primitive_types = []

    if params.primitive_type:
        with open(params.primitive_type_file, 'r', encoding='utf-8') as f:
            for line in f:
                primitive_types.append(line.strip())

    params.primitive_types = primitive_types

    print("get primitive_types:", primitive_types)

    primitive_type_ids = []

    for pt in primitive_types:
        primitive_type_ids.append(params.type2idx[pt])

    params.primitive_type_ids = primitive_type_ids

    print("get primitive_type_ids:", primitive_type_ids)

    test_tree_seqs = None

    if params.use_source_seq:
        test_trees, test_seqs, test_tree_seqs = prepare_data(params.test_file,
                                                             primitive_types=params.primitive_types,
                                                             tree_class=NaryTree,
                                                             use_type=params.use_type,
                                                             use_source_seq=params.use_source_seq,
                                                             use_copy=params.copy,
                                                             whatset="test",
                                                             save_input_to_file=params.save_input_to_file)
    else:
        test_trees, test_seqs = prepare_data(params.test_file,
                                             primitive_types=params.primitive_types,
                                             tree_class=NaryTree,
                                             use_type=params.use_type,
                                             use_source_seq=params.use_source_seq,
                                             use_copy=params.copy,
                                             whatset="test",
                                             save_input_to_file=params.save_input_to_file)

    if params.use_type:
        test_res = data2id_v2(test_trees, test_seqs, t_word2idx, s_word2idx, type2idx=type2idx, tree_seqs=test_tree_seqs, copy=params.copy, whatset="test")
    else:
        test_res = data2id_v2(test_trees, test_seqs, t_word2idx, s_word2idx, tree_seqs=test_tree_seqs, copy=params.copy, whatset="test")

    if params.copy:
        test_dataset = test_res[0]
        params.s_with_tree_in_seq_unk_word2idx = test_res[1]
        params.s_with_tree_in_seq_unk_vocab_size = len(test_res[1])

        s_with_tree_in_seq_unk_idx2word = {}

        for word, idx in params.s_with_tree_in_seq_unk_word2idx.items():
            s_with_tree_in_seq_unk_idx2word[idx] = word

        params.s_with_tree_in_seq_unk_idx2word = s_with_tree_in_seq_unk_idx2word

        print("add %d words unknown tree word in s_word2idx (testset): " % (len(params.s_with_tree_in_seq_unk_word2idx) - len(params.seq_word2idx)))
        print("seq_with_tree_in_seq_unk_vocab_size:", params.s_with_tree_in_seq_unk_vocab_size)

    else:
        test_dataset = test_res

    return test_dataset

def evaluate(sess, params, dataset, model, eos_id, metric_fn, best_eval_value):

    print("Evaluation------")

    sample_ids = []
    targets = []

    seq_in_data = []
    seq_out_data = []
    tree_data = []

    eval_total_loss = 0.0

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    while not dataset.is_end():
        batch = dataset.get_batch(batch_size=params.batch_size)

        sample_id, batch_eval_loss, batch_eval_total_loss = model.predict(sess=sess, batch=batch)

        eval_total_loss += batch_eval_total_loss

        sample_id = sample_id.tolist()

        sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in sample_id]

        # if not params.copy:
        # 
        #     sample_id = [[seq_idx2word[token_id] for token_id in sample] for sample in sample_id]
        # 
        # else:
        #     processed_sample_id = []
        # 
        #     for sample in sample_id:
        # 
        #         processed_sample = []
        # 
        #         for token_id in sample:
        # 
        #             processed_sample.append(seq_idx2word[token_id])
        # 
        #         processed_sample_id.append(processed_sample)
        # 
        #     sample_id = processed_sample_id

        sample_id, _ = post_processing(params, sample_id)

        sample_ids.extend(sample_id)

        target_id = []

        for sample in batch['batch_gold_seqs']:

            one_example_target_id = []

            for token in sample:
                one_example_target_id.extend(token.split(params.separator))

            target_id.append(one_example_target_id)

        targets.extend(target_id)

        batch_seq_in_data = [[seq_idx2word[token_id] for token_id in seq_in] for seq_in in batch['batch_seq_in_data']]
        seq_in_data.extend(batch_seq_in_data)

        batch_seq_out_data = [[seq_idx2word[token_id] for token_id in seq_out] for seq_out in batch['batch_seq_out_data']]
        seq_out_data.extend(batch_seq_out_data)

        batch_tree_data = [json.dumps([[params.t_idx2word[val_token_id] for val_token_id in node] for node in tree]) for tree in batch['batch_tree_data']]
        tree_data.extend(batch_tree_data)

    dataset.reset()

    eval_value = metric_fn(sample_ids, targets)

    if params.save_output_to_file is not None and (eval_value > best_eval_value):
        with open(params.save_output_to_file + ".evaluate", "w", encoding="utf-8") as output_f:
            for pred, gold, seq_in, seq_out, tree in zip(sample_ids, targets, seq_in_data, seq_out_data, tree_data):
                output_f.write("Predict: " + ' '.join(pred) + "\n")
                output_f.write("Gold: " + ' '.join(gold) + "\n")
                output_f.write("Tree: " + tree + "\n")
                output_f.write("Seq In: " + ' '.join(seq_in) + "\n")
                output_f.write("Seq Out: " + ' '.join(seq_out) + "\n\n")

    return eval_value, eval_total_loss



def train(params, train_dataset, dev_dataset):
    t_word2idx = params.tree_word2idx
    s_word2idx = params.seq_word2idx

    pad_id = t_word2idx[params.pad_token]
    unk_id = t_word2idx[params.unk_token]

    s_idx2word = {}

    for w in s_word2idx:
        s_idx2word[s_word2idx[w]] = w


    model = Tree2SeqModel(params)

    model.build_graph()

    num_epoch = params.num_epoch

    load_model = params.load_model

    train_step = 0

    # collect step-level training total loss
    step_train_total_losses = []

    # collect epoch-level training total loss
    epoch_train_total_losses = []

    # collect epoch-level training metric(bleu or rouge)
    epoch_train_metrics = []

    # collect step-level evaluation total loss
    step_eval_total_losses = []

    # collect step-level evaluation metric(bleu or rouge)
    step_eval_metrics = []

    # collect epoch-level evaluation total loss
    epoch_eval_total_losses = []

    # collect epoch-level evaluation metric
    epoch_eval_metrics = []

    # primitive_type_ids = []
    # 
    # if params.primitive_type:
    #     with open(params.primitive_type_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             primitive_type_ids.append(params.type2idx[line.strip()])

    train_dataset = Dataset_V2(train_dataset, max_child_num=params.max_child_num, shuffle=True, use_source_seq=params.use_source_seq, pad_id=pad_id, unk_id=unk_id, pad_token=params.pad_token, unk_token=params.unk_token, copy=params.copy, name="trainset", primitive_type_ids=params.primitive_type_ids, copy_once=params.copy_once)
    dev_dataset =  Dataset_V2(dev_dataset, max_child_num=params.max_child_num, use_source_seq=params.use_source_seq, pad_id=pad_id, unk_id=unk_id, pad_token=params.pad_token, unk_token=params.unk_token, copy=params.copy, name="devset", primitive_type_ids=params.primitive_type_ids, copy_once=params.copy_once)

    # train_valid_node_counts = []
    # train_tree_node_counts = []
    # while not train_dataset.is_end():
    #     one = train_dataset.get_batch(batch_size=1)
    #     train_valid_node_counts.append(sum(one['batch_encoder_mask'][0]))
    #     train_tree_node_counts.append(one['batch_tree_node_counts'][0])
    #
    # dev_valid_node_counts = []
    # dev_tree_node_counts = []
    #
    # while not dev_dataset.is_end():
    #     one = dev_dataset.get_batch(batch_size=1)
    #     dev_valid_node_counts.append(sum(one['batch_encoder_mask'][0]))
    #     dev_tree_node_counts.append(one['batch_tree_node_counts'][0])
    #
    # print("train avg valid node count:", sum(train_valid_node_counts) / len(train_valid_node_counts))
    # print("train avg tree node count:", sum(train_tree_node_counts) / len(train_tree_node_counts))
    #
    # print("dev avg valid node count:", sum(dev_valid_node_counts) / len(dev_valid_node_counts))
    # print("dev avg tree node count:", sum(dev_tree_node_counts) / len(dev_tree_node_counts))
    #
    # exit(0)

    eos_id = s_word2idx[params.end_token]

    train_sample_ids = []
    train_targets = []

    saver = tf.train.Saver()

    checkpoint_prefix = os.path.join(params.checkpoint_dir, "model")

    rl_restore_checkpoint_prefix = os.path.join(params.rl_restore_checkpoint_dir, "model")

    for param in tf.trainable_variables():
        print(param)

    best_dev_metric = 0

    cur_dev_metric = 0

    pretrain_dev_metric = -1

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    if params.metric == "bleu":
        metric_fn = bleu_fn
    elif params.metric == "rouge":
        metric_fn = rouge_fn
    else:
        raise ValueError('no evaluation metric: %s' % (params.metric))

    with tf.Session(config=config) as sess:

      # just used when classifier is 'rl'
        if params.classifier == "rl":
            reward_sess = tf.Session(config=config, graph=tf.get_default_graph())

        if load_model:
            # restore parameters
            saver.restore(sess=sess, save_path=checkpoint_prefix)
            print("Restore Model from %s...." % (checkpoint_prefix))

            if params.classifier == "rl":
                saver.save(sess, save_path=rl_restore_checkpoint_prefix)
        else:
            sess.run(tf.global_variables_initializer())

            if params.classifier == "rl":
                saver.save(sess, save_path=rl_restore_checkpoint_prefix)

        for e in range(num_epoch):
            print("=========================")
            print("=========================")
            print("=========================")
            print("\n\n\n\n\n")
            train_total_loss = 0.0

            while not train_dataset.is_end():

                train_step += 1

                batch = train_dataset.get_batch(batch_size=params.batch_size)

                r_sess = None
                r_saver = None
                r_checkpoint_prefix = None

                if params.classifier == "rl":
                    if train_step > params.pretrain_steps:
                        print("RL Training Start !")

                    batch['rl_train'] = (train_step > params.pretrain_steps)
                    r_sess = reward_sess
                    r_saver = saver
                    r_checkpoint_prefix = rl_restore_checkpoint_prefix


                batch_loss, batch_train_total_loss, sample_id = model.train(sess=sess, batch=batch, reward_sess=r_sess, saver=r_saver, checkpoint_prefix=r_checkpoint_prefix)

                train_total_loss += batch_train_total_loss

                print("Epoch%d-train_step%d-loss=%f"%(e+1, train_step, batch_loss))
                print("current dev bleu:", cur_dev_metric)
                print("pretrain dev bleu:", pretrain_dev_metric)
                print("l2 weight:", params.l2_weight, " best dev bleu:", best_dev_metric)
                print("****************************************************************")
                print("\n")
                # evaluate during one training epoch
                if (train_step > params.eval_begin_step and train_step % params.eval_every_steps == 0) or (train_step == params.pretrain_steps):
                    eval_value, cur_eval_total_loss = evaluate(sess=sess, params=params, dataset=dev_dataset, model=model, eos_id=eos_id, metric_fn=metric_fn, best_eval_value=best_dev_metric)
                    print("Epoch", e + 1, "Dev Bleu is:", eval_value)
                    step_eval_metrics.append((train_step, eval_value))
                    step_eval_total_losses.append((train_step, cur_eval_total_loss))

                    step_train_total_losses.append((train_step, train_total_loss))

                    cur_dev_metric = eval_value

                    if train_step == params.pretrain_steps and params.classifier == "rl" and pretrain_dev_metric < 0:
                        pretrain_dev_metric = eval_value

                    with open(params.eval_result_log, "a+", encoding="utf-8") as log_file:
                        log_file.write("Epoch" + str(e + 1) + "-Dev Bleu: " + str(eval_value))

                    if eval_value > best_dev_metric:
                        best_dev_metric = eval_value
                        saver.save(sess, save_path=checkpoint_prefix)

                        print("Save a new model as %s !" % (checkpoint_prefix))

                    print("best dev bleu:", best_dev_metric)

                train_sample_id = sample_id.tolist()

                train_sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in train_sample_id]

                train_sample_id = [[seq_idx2word[token_id] for token_id in sample] for sample in train_sample_id]

                train_sample_ids.extend(train_sample_id)

                # target_id = [tgt[:tgt.index(eos_id)] for tgt in batch['batch_seq_out_data']]

                # target_id = [[params.s_idx2word[token_id] for token_id in sample] for sample in target_id]

                target_id = batch['batch_gold_seqs']

                train_targets.extend(target_id)

            train_bleu = metric_fn(train_sample_ids, train_targets)
            print("train bleu:", train_bleu)
            epoch_train_metrics.append((e+1, train_bleu))
            epoch_train_total_losses.append((e+1, train_total_loss))

            train_sample_ids = []
            train_targets = []

            train_dataset.reset()

            print("Finish A Epoch Training !")
            
            eval_value, cur_epoch_eval_total_loss = evaluate(sess=sess, params=params, dataset=dev_dataset, model=model, eos_id=eos_id, metric_fn=metric_fn, best_eval_value=best_dev_metric)
            epoch_eval_metrics.append((e+1, eval_value))
            epoch_eval_total_losses.append((e+1, cur_epoch_eval_total_loss))

            print("Epoch", e+1, "Dev Bleu is:", eval_value)

            cur_dev_metric = eval_value

            if train_step == params.pretrain_steps and params.classifier == "rl" and pretrain_dev_metric < 0:
                pretrain_dev_metric = eval_value

            with open(params.eval_result_log, "a+", encoding="utf-8") as log_file:
                log_file.write("Epoch" + str(e+1) + "-Dev Bleu: " + str(eval_value))

            if eval_value > best_dev_metric:
                best_dev_metric = eval_value
                saver.save(sess, save_path=checkpoint_prefix)

                print("Save a new model as %s !" % (checkpoint_prefix))

            print("best dev bleu:", best_dev_metric)

                # sample_sentences = [' '.join([s_idx2word[wid] for wid in ids]) for ids in sample_ids]
                #
                # with open(params.predict_file, "w", encoding="utf-8") as predict_f:
                #
                #     for line in sample_sentences:
                #         predict_f.write(line + "\n")

        if params.classifier == "rl":
            reward_sess.close()

        infer_eval_dict = {}
        infer_eval_dict['step_train_total_losses'] = step_train_total_losses
        infer_eval_dict['epoch_train_total_losses'] = epoch_train_total_losses
        infer_eval_dict['epoch_train_metrics'] = epoch_train_metrics
        infer_eval_dict['step_eval_total_losses'] = step_eval_total_losses
        infer_eval_dict['step_eval_metrics'] = step_eval_metrics
        infer_eval_dict['epoch_eval_total_losses'] = epoch_eval_total_losses
        infer_eval_dict['epoch_eval_metrics'] = epoch_eval_metrics

        with open(params.iter_eval_log, 'w', encoding='utf-8') as f:
            json.dump(infer_eval_dict, f)


def test(params, test_dataset):

    t_word2idx = params.tree_word2idx
    s_word2idx = params.seq_word2idx

    pad_id = t_word2idx[params.pad_token]
    unk_id = t_word2idx[params.unk_token]

    model = Tree2SeqModel(params)
    model.build_graph()

    # primitive_type_ids = []
    #
    # if params.primitive_type:
    #     with open(params.primitive_type_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             primitive_type_ids.append(params.type2idx[line.strip()])

    test_dataset = Dataset_V2(test_dataset, max_child_num=params.max_child_num, use_source_seq=params.use_source_seq, pad_id=pad_id, unk_id=unk_id, pad_token=params.pad_token, unk_token=params.unk_token, copy=params.copy, name="testset", primitive_type_ids=params.primitive_type_ids, copy_once=params.copy_once)

    eos_id = s_word2idx[params.end_token]

    saver = tf.train.Saver()
    checkpoint_prefix = os.path.join(params.checkpoint_dir, "model")

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    if params.metric == "bleu":
        metric_fn = bleu_fn
    elif params.metric == "rouge":
        metric_fn = rouge_fn
    else:
        raise ValueError('no evaluation metric: %s' % (params.metric))

    with tf.Session(config=config) as sess:

        if params.classifier == "rl":
            reward_sess = tf.Session(config=config, graph=tf.get_default_graph())
            saver.restore(sess=reward_sess, save_path=checkpoint_prefix)

        saver.restore(sess=sess, save_path=checkpoint_prefix)

        sample_ids = []
        targets = []

        seq_in_data = []
        seq_out_data = []
        tree_data = []

        test_total_loss = 0.0

        while not test_dataset.is_end():
            test_batch = test_dataset.get_batch(batch_size=params.batch_size)

            test_sample_id, batch_test_loss, batch_test_total_loss = model.predict(sess=sess, batch=test_batch)

            test_total_loss += batch_test_total_loss

            test_sample_id = test_sample_id.tolist()

            test_sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in test_sample_id]

            # if not params.copy:
            #
            #     test_sample_id = [[seq_idx2word[token_id] for token_id in sample] for sample in test_sample_id]
            #
            # else:
            #     processed_sample_id = []
            #
            #     for sample in test_sample_id:
            #
            #         processed_sample = []
            #
            #         for token_id in sample:
            #             # processed_sample.extend(seq_idx2word[token_id].split(params.separator))
            #             processed_sample.append(seq_idx2word[token_id])
            #
            #         processed_sample_id.append(processed_sample)
            #
            #     test_sample_id = processed_sample_id

            test_sample_id, _ = post_processing(params, test_sample_id)

            sample_ids.extend(test_sample_id)

            # target_id = [tgt[:tgt.index(eos_id)] for tgt in test_batch['batch_seq_out_data']]

            # target_id = [[params.s_idx2word[token_id] for token_id in sample] for sample in target_id]

            # target_id = test_batch['batch_gold_seqs']

            target_id = []

            for sample in test_batch['batch_gold_seqs']:

                one_example_target_id = []

                for token in sample:
                    one_example_target_id.extend(token.split(params.separator))

                target_id.append(one_example_target_id)

            targets.extend(target_id)

            batch_seq_in_data = [[seq_idx2word[token_id] for token_id in seq_in] for seq_in in test_batch['batch_seq_in_data']]
            seq_in_data.extend(batch_seq_in_data)

            batch_seq_out_data = [[seq_idx2word[token_id] for token_id in seq_out] for seq_out in test_batch['batch_seq_out_data']]
            seq_out_data.extend(batch_seq_out_data)

            batch_tree_data = [json.dumps([[params.t_idx2word[val_token_id] for val_token_id in node] for node in tree]) for tree in test_batch['batch_tree_data']]
            tree_data.extend(batch_tree_data)

        test_dataset.reset()

        test_value = metric_fn(sample_ids, targets)

        print("Test BLEU:", test_value)

        if params.classifier == "rl":
            reward_sess.close()

        if params.save_output_to_file is not None:
            with open(params.save_output_to_file + ".test", "w", encoding="utf-8") as output_f:
                for pred, gold, seq_in, seq_out, tree in zip(sample_ids, targets, seq_in_data, seq_out_data, tree_data):
                    output_f.write("Predict: " + ' '.join(pred) + "\n")
                    output_f.write("Gold: " + ' '.join(gold) + "\n")
                    output_f.write("Tree: " + tree + "\n")
                    output_f.write("Seq In: " + ' '.join(seq_in) + "\n")
                    output_f.write("Seq Out: " + ' '.join(seq_out) + "\n\n")



# def distributed_train(params, train_dataset, dev_dataset):
#     t_word2idx = params.tree_word2idx
#     s_word2idx = params.seq_word2idx
#
#     gpu_checker = GPUChecker()
#
#     num_gpus = gpu_checker.get_gpu_num()
#
#     distribution = tf.contrib.distribute.MirroredStrategy(devices=None, num_gpus=num_gpus)
#     # distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=num_gpus)
#
#     # global config
#     # distribution.configure(session_config=config)
#
#     with distribution.scope():
#
#         model = Tree2SeqModel(params)
#
#         num_epoch = params.num_epoch
#
#         train_step = 0
#
#         train_dataset = Dataset(train_dataset, shuffle=True)
#
#         dev_dataset = Dataset(dev_dataset)
#
#         eos_id = s_word2idx[params.end_token]
#
#         for e in range(num_epoch):
#
#             while not train_dataset.is_end():
#                 train_step += 1
#
#                 batch = train_dataset.get_batch(batch_size=params.batch_size*num_gpus)
#
#                 def training_step(gpu_num_batch, each_gpu_batch_size, total_size):
#                     tower_ctx = tf.contrib.distribute.get_tower_context()
#
#                     start = tower_ctx.tower_id * each_gpu_batch_size
#
#                     if start >= total_size:
#                         print("total_size is %d less than device %s (tower id is %d) 's  start index %d"%(total_size, tower_ctx.device, tower_ctx.tower_id, start))
#                         return tf.constant(-1, dtype=tf.float32)
#
#                     end = min((tower_ctx.tower_id+1)*each_gpu_batch_size, total_size)
#
#                     # print("get_batch:%d-%d"%(start, end))
#
#                     loss, sample_id = model.train(encoder_inputs=gpu_num_batch['batch_tree_data'][start:end],
#                                               node_lens=gpu_num_batch['batch_node_lens'][start:end],
#                                               parent_child_tables=gpu_num_batch['batch_parent_child_tables'][start:end],
#                                               decoder_inputs=gpu_num_batch['batch_seq_in_data'][start:end],
#                                               decoder_input_lengths=gpu_num_batch['batch_seq_lens'][start:end],
#                                               targets=gpu_num_batch['batch_seq_out_data'][start:end],
#                                               parent_child_types=gpu_num_batch['batch_parent_child_types'][start:end],
#                                               tree_seq_in_ids=gpu_num_batch['batch_tree_seq_in_data'],
#                                               tree_seq_lens=gpu_num_batch['batch_tree_seq_lens'])
#
#                     return loss
#
#
#                 total_size = len(batch['batch_tree_data'])
#
#                 each_gpu_batch_size = math.ceil(float(total_size) / float(num_gpus))
#
#                 merged_results = distribution.call_for_each_tower(training_step, batch, each_gpu_batch_size, total_size, run_concurrently=model.built())
#
#                 print("Epoch%d-train_step%d-loss=%f" % (e, train_step, distribution.unwrap(merged_results)[0]))
#
#             train_dataset.reset()
#
#             print("Evaluation------")
#
#             sample_ids = []
#             targets = []
#             while not dev_dataset.is_end():
#                 dev_batch = dev_dataset.get_batch(batch_size=params.batch_size)
#
#                 dev_sample_id = model.predict(encoder_inputs=dev_batch['batch_tree_data'],
#                                               node_lens=dev_batch['batch_node_lens'],
#                                               parent_child_tables=dev_batch['batch_parent_child_tables'],
#                                               parent_child_types=dev_batch['batch_parent_child_types'],
#                                               tree_seq_in_ids=dev_batch['batch_tree_seq_in_data'],
#                                               tree_seq_lens=dev_batch['batch_tree_seq_lens'])
#
#                 dev_sample_id = dev_sample_id.numpy().tolist()
#
#                 dev_sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in
#                                  dev_sample_id]
#
#                 sample_ids.extend(dev_sample_id)
#
#                 target_id = [tgt[:tgt.index(eos_id)] for tgt in dev_batch['batch_seq_out_data']]
#
#                 targets.extend(target_id)
#
#             dev_dataset.reset()
#
#             eval_value = bleu_fn(sample_ids, targets)
#
#             print("Epoch", e, "Dev Bleu is:", eval_value)

if __name__== '__main__':

    parser = argparse.ArgumentParser()

    # data_config = get_data_config('conala', trainset='train', devset='dev', testset='test')
    # data_config = get_data_config('atis', trainset='train', devset='dev', testset='test')
    data_config = get_data_config('wikisql', trainset='train', devset='dev', testset='test')

    add_arguments(parser, data_config)
    params = parser.parse_args()

    params.start_token = data_utils.SOS
    params.end_token = data_utils.EOS
    params.pad_token = data_utils.PAD
    params.unk_token = data_utils.UNK
    params.separator = data_utils.SEP

    tf.random.set_random_seed(params.rand_seed)

    if params.mode == 'train':
        print("Training Mode Starts !")
        train_dataset, dev_dataset = prepare_train_data(params)

        print("src vocab size:", params.src_vocab_size)
        print("tgt vocab size:", params.tgt_vocab_size)

        if not params.distributed:
            train(params, train_dataset, dev_dataset)
        else:
            raise Exception('not support distributed training now !')
            # distributed_train(params, train_dataset, dev_dataset)
    elif params.mode == 'test':
        print("Testing Mode Starts !")
        test_dataset = prepare_test_data(params)
        print("src vocab size:", params.src_vocab_size)
        print("tgt vocab size:", params.tgt_vocab_size)

        test(params, test_dataset)
    else:
        raise Exception("the Argument 'mode' must be 'train' or 'test'.")

