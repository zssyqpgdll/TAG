import json
# import sys
import os
# print(sys.path)
from treelstm.tree import BinaryTree, NaryTree

from copy import deepcopy

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SEP = "[SEP]"

def prepare_data(file_path,
                 primitive_types,
                 tree_class=BinaryTree,
                 format='json',
                 use_type=False,
                 use_source_seq=False,
                 use_copy=False,
                 whatset="train",
                 save_input_to_file=None):
    # if use copy, we use single word in train & test tree data (source), and replace corresponding tree node word into
    # train seq's word which is contained by the tree node word

    trees = []

    seqs = []

    source_seqs = []

    seq_lens = []

    if format == 'json':

        js = json.load(open(file_path, "r", encoding="utf-8"))

        print(len(js))

        replace_counts = []

        for j in js:
            # print("json:", j)
            tree = tree_class(j['source_ast'], primitive_types=primitive_types, use_type=use_type, single_word=use_copy, separator=SEP)
            trees.append(tree)

            target_seq_words = j['target_prog']

            if use_copy:
                replace_flag_list = [False] * len(target_seq_words)

                target_seq_words, replace_count, _ = tree.process_sub_word_in_seq(target_seq_words, replace_flag_list)
                replace_counts.append(replace_count)
                # pass

            seq_lens.append(len(target_seq_words))
            seqs.append(' '.join(target_seq_words))

            if use_source_seq:
                source_seqs.append(' '.join(j['source_prog']))

        print(whatset + " set-max-len:", max(seq_lens))
    else:
        raise Exception('the format:%s is invalid'%(format))

    if use_copy:
        print("in %sset target seq replaced word count max: %d"%(whatset, max(replace_counts)))
        print("in %sset target seq replaced word count min: %d"%(whatset, min(replace_counts)))
        # pass

    if save_input_to_file is not None:
        print("saving input data to file:" + save_input_to_file)
        save_input_data_to_file(save_input_to_file, trees, seqs, whatset)

    if use_source_seq:
        return trees, seqs, source_seqs

    return trees, seqs


def save_input_data_to_file(save_file, trees, seqs, whatset):
    # 获取文件夹路径
    folder_path = os.path.dirname(save_file)

    # 如果文件夹不存在，创建文件夹
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(save_file + "." + whatset, "w", encoding="utf-8") as save_f:

        for tree, seq in zip(trees, seqs):
            tree_json = json.dumps(tree.to_serializable())
            seq_json = json.dumps(seq)

            save_f.write(tree_json + "\t" + seq_json + "\n")



def get_tree_vocab(vocab, trees, add_special_tokens=True, tree_seqs=None, min_vocab_count=None):
    '''
    :param vocab: a dict
    :param trees: list of BinaryTree
    :return: None
    '''

    if add_special_tokens:
        vocab[PAD] = 0
        vocab[SOS] = 1
        vocab[EOS] = 2
        vocab[UNK] = 3

    vocab2freq = {}

    for t in trees:
        t.get_vocab(vocab2freq)

    if tree_seqs is not None:
        for s in tree_seqs:
            for w in s.strip().split():
                if w not in vocab2freq:
                    vocab2freq[w] = 1
                else:
                    vocab2freq[w] += 1


    if min_vocab_count is not None:

        for v in vocab2freq:

            if vocab2freq[v] >= min_vocab_count:

                vocab[v] = len(vocab)


def get_seq_vocab(vocab, seqs, add_special_tokens=True, min_vocab_count=None):
    '''
    :param vocab: a list
    :param seqs:  list of sequence
    :return: None
    '''

    if add_special_tokens:
        vocab[PAD] = 0
        vocab[SOS] = 1
        vocab[EOS] = 2
        vocab[UNK] = 3

    vocab2freq = {}

    for s in seqs:
        for w in s.strip().split():

            if w not in vocab2freq:
                vocab2freq[w] = 1
            else:
                vocab2freq[w] += 1

    if min_vocab_count is not None:

        for v in vocab2freq:

            if vocab2freq[v] >= min_vocab_count:

                vocab[v] = len(vocab)

# def get_type_vocab(vocab_path, limit=None):
#
#     d = {}
#
#     with open(vocab_path, "r", encoding="utf-8") as vocab_f:
#
#         for line in vocab_f:
#             line = line.strip()
#             split = line.split()
#             tp = split[0]
#             freq = int(split[1])
#             d[tp] = freq
#
#     tmp_vocab = sorted(d, key=lambda x:d[x], reverse=True)
#
#     if limit is not None:
#         tmp_vocab = tmp_vocab[:limit]
#
#     vocab = {"<unk>":0}
#
#     for v in tmp_vocab:
#         vocab[v] = len(vocab)
#
#     return vocab


def get_type_vocab(vocab, trees, limit=None):

    vocab_freqs = {}

    for t in trees:
        t.get_type_vocab(vocab_freqs)

    tmp_vocab = sorted(vocab_freqs, key=lambda x: vocab_freqs[x], reverse=True)

    if limit is not None:
        tmp_vocab = tmp_vocab[:limit]

    vocab['OTHER'] = 0

    for v in tmp_vocab:
        if v not in vocab:
            vocab[v] = len(vocab)



def data2id(trees, seqs, t_word2idx, s_word2idx, type2idx=None, tree_seqs=None, copy=False):


    tree_data = []
    node_lens = []
    # parent_child_tables = []
    left_child_ids = []
    right_child_ids = []

    tmp_seq_in_data = []
    tmp_seq_out_data = []
    seq_lens = []

    tmp_tree_seq_in_data = []
    tmp_tree_seq_out_data = []
    tree_seq_lens = []

    # parent_child_types = []
    parent_types = []
    left_child_types = []
    right_child_types = []

    tree_node_counts = []

    gold_seqs = []

    tree2nl_ids = []

    use_type = (type2idx is not None)

    for index, (t, s) in enumerate(zip(trees, seqs)):

            queue = [t]

            node_ids = []
            tree2nl_node_ids = []

            node_id_lens = []

            # parent_child_table = []
            lc_ids = []
            rc_ids = []

            # parent_child_type = []
            p_types = []
            lc_types = []
            rc_types = []

            value2id = {}

            while len(queue) > 0:
                node = queue.pop(0)

                node_id = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in node.value.strip().split()]

                node_ids.append(node_id)

                tree2nl_node_id = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in node.value.strip().split()]

                tree2nl_node_ids.append(tree2nl_node_id)

                # value2id[node.value] = len(tmp_node_ids) - 1
                value2id[node] = len(node_ids) - 1

                node_id_lens.append(len(node_id))

                parent_type = 0
                if use_type:
                    parent_type = type2idx[node.type] if node.type in type2idx else 0

                left_child_value = None
                left_child_type = 0     # OTHER type
                if node.left_child is not None:
                    queue.append(node.left_child)
                    # left_child_value = node.left_child.value
                    left_child_value = node.left_child

                    if use_type:
                        left_child_type = type2idx[node.left_child.type] if node.left_child.type in type2idx else 0

                right_child_value = None
                right_child_type = 0      # OTHER type
                if node.right_child is not None:
                    queue.append(node.right_child)
                    # right_child_value = node.right_child.value
                    right_child_value = node.right_child

                    if use_type:
                        right_child_type = type2idx[node.right_child.type] if node.right_child.type in type2idx else 0

                # parent_child_table.append([left_child_value, right_child_value])
                lc_ids.append(left_child_value)
                rc_ids.append(right_child_value)

                p_types.append(parent_type)
                lc_types.append(left_child_type)
                rc_types.append(right_child_type)
                    # parent_child_type.append((parent_type, left_child_type, right_child_type))

            # value2id[PAD] = len(tmp_node_ids)

            for i in range(len(lc_ids)):

                if lc_ids[i] is not None:
                    lc_ids[i] = value2id[lc_ids[i]]
                else:
                    lc_ids[i] = len(value2id)

                if rc_ids[i] is not None:
                    rc_ids[i] = value2id[rc_ids[i]]
                else:
                    rc_ids[i] = len(value2id)

            left_child_ids.append(lc_ids)
            right_child_ids.append(rc_ids)

            # parent_child_types.append(parent_child_type)
            parent_types.append(p_types)
            left_child_types.append(lc_types)
            right_child_types.append(rc_types)

            # pad_id = t_word2idx[PAD]

            # max_node_id_len = max(node_id_lens)

            # node_ids = []
            # node_ids.extend(tmp_node_ids)
            # for node_id, node_id_len in zip(tmp_node_ids, node_id_lens):
            #     node_ids.append(node_id + [pad_id]*(max_node_id_len - node_id_len))

            tree_node_counts.append(len(node_ids))
            tree_data.append(node_ids)

            tree2nl_ids.append(tree2nl_node_ids)

            node_lens.append(node_id_lens)

            s_in = [SOS] + s.strip().split()
            s_out = s.strip().split() + [EOS]
            gold_seq = s.strip().split()

            gold_seqs.append(gold_seq)

            seq_in_ids = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in s_in]
            seq_out_ids = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in s_out]
            seq_lens.append(len(seq_in_ids))
            tmp_seq_in_data.append(seq_in_ids)
            tmp_seq_out_data.append(seq_out_ids)

            if tree_seqs is not None:
                tree_s = tree_seqs[index]
                tree_s_in = [SOS] + tree_s.strip().split()
                tree_s_out = tree_s.strip().split() + [EOS]

                tree_seq_in_ids = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in tree_s_in]
                tree_seq_out_ids = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in tree_s_out]
                tree_seq_lens.append(len(tree_seq_in_ids))
                tmp_tree_seq_in_data.append(tree_seq_in_ids)
                tmp_tree_seq_out_data.append(tree_seq_out_ids)

    # max_seq_len = max(seq_lens)

    pad_id = s_word2idx[PAD]

    seq_in_data = []
    seq_out_data = []

    seq_in_data.extend(tmp_seq_in_data)
    seq_out_data.extend(tmp_seq_out_data)

    # for s_in, s_out, l in zip(tmp_seq_in_data, tmp_seq_out_data, seq_lens):
    #     seq_in_data.append(s_in + [pad_id]*(max_seq_len - l))
    #     seq_out_data.append(s_out + [pad_id]*(max_seq_len - l))


    dataset  = {}
    dataset['tree_data'] = tree_data
    dataset['seq_in_data'] = seq_in_data
    dataset['seq_out_data'] = seq_out_data
    dataset['node_lens'] = node_lens
    dataset['seq_lens'] = seq_lens
    # dataset['parent_child_tables'] = parent_child_tables
    # dataset['parent_child_types'] = parent_child_types

    dataset['left_child_ids'] = left_child_ids
    dataset['right_child_ids'] = right_child_ids

    dataset['parent_types'] = parent_types
    dataset['left_child_types'] = left_child_types
    dataset['right_child_types'] = right_child_types
    dataset['tree_node_counts'] = tree_node_counts
    dataset['gold_seqs'] = gold_seqs

    if tree_seqs is not None:

        max_tree_seq_len = max(tree_seq_lens)

        tree_seq_in_data = []
        tree_seq_out_data = []

        for tree_s_in, tree_s_out, l in zip(tmp_tree_seq_in_data, tmp_tree_seq_out_data, tree_seq_lens):

            tree_seq_in_data.append(tree_s_in + [pad_id]*(max_tree_seq_len - l))
            tree_seq_out_data.append(tree_s_out + [pad_id]*(max_tree_seq_len - l))

        dataset['tree_seq_in_data'] = tree_seq_in_data
        dataset['tree_seq_out_data'] = tree_seq_out_data
        dataset['tree_seq_lens'] = tree_seq_lens

    if copy:
        dataset['tree2nl_ids'] = tree2nl_ids

    return dataset

def data2id_v2(trees, seqs, t_word2idx, s_word2idx, type2idx=None, tree_seqs=None, copy=False, whatset="train"):
    tree_data = []
    node_lens = []

    tree2nl_ids = []

    tmp_seq_in_data = []
    tmp_seq_out_data = []
    seq_lens = []
    gold_seqs = []

    tmp_tree_seq_in_data = []
    tmp_tree_seq_out_data = []
    tree_seq_lens = []

    child_ids = []

    parent_types = []

    child_types = []


    tree_node_counts = []

    tree_data_tokens = []
    seq_out_tokens = []

    s_with_tree_in_seq_unk_word2idx = deepcopy(s_word2idx)

    use_type = (type2idx is not None)

    for index, (t, s) in enumerate(zip(trees, seqs)):

        queue = [t]

        node_ids = []
        node_tokens = []

        node_id_lens = []

        tree2nl_node_ids = []

        p_types = []

        one_example_child_ids = []
        one_example_child_types = []

        value2id = {}

        while len(queue) > 0:

            node = queue.pop(0)

            node_id = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in node.value.strip().split()]

            if copy:
                assert len(node_id) <= 1, "node value length is > 1"

            node_ids.append(node_id)

            node_token = [w for w in node.value.strip().split()]
            node_tokens.append(node_token)

            if whatset == "train":

                # used in non-train mode and train-rl mode, otherwise is just redutant
                tree2nl_node_id = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in node.value.strip().split()]

            else:
                tree2nl_node_id = []

                # for w in node.value.strip().split():
                #
                #     if w not in s_with_tree_in_seq_unk_word2idx:
                #         s_with_tree_in_seq_unk_word2idx[w] = len(s_with_tree_in_seq_unk_word2idx)
                #
                #     tree2nl_node_id.append(s_with_tree_in_seq_unk_word2idx[w])

                concat_node_value = node.separator.join(node.value.strip().split())

                if concat_node_value not in s_with_tree_in_seq_unk_word2idx:
                    s_with_tree_in_seq_unk_word2idx[concat_node_value] = len(s_with_tree_in_seq_unk_word2idx)

                tree2nl_node_id.append(s_with_tree_in_seq_unk_word2idx[concat_node_value])

                assert len(tree2nl_node_id) == 1, "when use copy(pointer network), tree2nl_node_id length must equal 1 !"

            tree2nl_node_ids.append(tree2nl_node_id)

            # value2id[node.value] = len(tmp_node_ids) - 1
            value2id[node] = len(node_ids) - 1

            node_id_lens.append(len(node_id))

            parent_type = 0
            if use_type:
                parent_type = type2idx[node.type] if node.type in type2idx else 0

            p_types.append(parent_type)

            one_node_child_ids = []
            one_node_child_types = []

            for child_tree in node.children:
                queue.append(child_tree)
                one_node_child_ids.append(child_tree)

                if use_type:
                    ch_type = type2idx[child_tree.type] if child_tree.type in type2idx else 0 # 0 means OTHER TYPE
                    one_node_child_types.append(ch_type)

            one_example_child_ids.append(one_node_child_ids)
            one_example_child_types.append(one_node_child_types)

        for i in range(len(one_example_child_ids)):

            for j in range(len(one_example_child_ids[i])):
                one_example_child_ids[i][j] = value2id[one_example_child_ids[i][j]]

        child_ids.append(one_example_child_ids)

        parent_types.append(p_types)

        child_types.append(one_example_child_types)

        # node_ids = []
        # node_ids.extend(tmp_node_ids)

        tree_node_counts.append(len(node_ids))
        tree_data.append(node_ids)
        tree_data_tokens.append(node_tokens)

        tree2nl_ids.append(tree2nl_node_ids)

        node_lens.append(node_id_lens)

        s_in = [SOS] + s.strip().split()
        s_out = s.strip().split() + [EOS]
        gold_seq = s.strip().split()

        gold_seqs.append(gold_seq)

        seq_in_ids = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in s_in]
        seq_out_ids = [s_word2idx[w] if w in s_word2idx else s_word2idx[UNK] for w in s_out]
        seq_lens.append(len(seq_in_ids))
        tmp_seq_in_data.append(seq_in_ids)
        tmp_seq_out_data.append(seq_out_ids)
        seq_out_tokens.append(s_out)

        if tree_seqs is not None:
            tree_s = tree_seqs[index]
            tree_s_in = [SOS] + tree_s.strip().split()
            tree_s_out = tree_s.strip().split() + [EOS]

            tree_seq_in_ids = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in tree_s_in]
            tree_seq_out_ids = [t_word2idx[w] if w in t_word2idx else t_word2idx[UNK] for w in tree_s_out]
            tree_seq_lens.append(len(tree_seq_in_ids))
            tmp_tree_seq_in_data.append(tree_seq_in_ids)
            tmp_tree_seq_out_data.append(tree_seq_out_ids)

    # max_seq_len = max(seq_lens)

    pad_id = s_word2idx[PAD]

    seq_in_data = []
    seq_out_data = []

    seq_in_data.extend(tmp_seq_in_data)
    seq_out_data.extend(tmp_seq_out_data)

    dataset = {}
    dataset['tree_data'] = tree_data
    dataset['seq_in_data'] = seq_in_data
    dataset['seq_out_data'] = seq_out_data
    dataset['node_lens'] = node_lens
    dataset['seq_lens'] = seq_lens

    dataset['child_ids'] = child_ids

    dataset['parent_types'] = parent_types
    dataset['child_types'] = child_types
    dataset['tree_node_counts'] = tree_node_counts
    dataset['gold_seqs'] = gold_seqs
    dataset['tree_data_tokens'] = tree_data_tokens
    dataset['seq_out_tokens'] = seq_out_tokens

    if tree_seqs is not None:

        max_tree_seq_len = max(tree_seq_lens)

        tree_seq_in_data = []
        tree_seq_out_data = []

        for tree_s_in, tree_s_out, l in zip(tmp_tree_seq_in_data, tmp_tree_seq_out_data, tree_seq_lens):
            tree_seq_in_data.append(tree_s_in + [pad_id] * (max_tree_seq_len - l))
            tree_seq_out_data.append(tree_s_out + [pad_id] * (max_tree_seq_len - l))

        dataset['tree_seq_in_data'] = tree_seq_in_data
        dataset['tree_seq_out_data'] = tree_seq_out_data
        dataset['tree_seq_lens'] = tree_seq_lens

    if copy:
        dataset['tree2nl_ids'] = tree2nl_ids

        if whatset != "train":
            return dataset, s_with_tree_in_seq_unk_word2idx

    return dataset


def post_processing(params, batch_example_ids):

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    unk_id = params.tree_word2idx[params.unk_token]

    pad_id = params.tree_word2idx[params.pad_token]

    tgt_vocab_size = params.tgt_vocab_size

    separator = params.separator

    new_batch_example_ids = []

    for example in batch_example_ids:

        new_example = []

        before_multitoken_idx = -1

        for i, word_idx in enumerate(example):

            # if word_idx >= tgt_vocab_size:
            #
            #     if before_multitoken_idx < 0:
            #         before_multitoken_idx = word_idx
            #     else:
            #         if before_multitoken_idx != word_idx:
            #             new_example.append(before_multitoken_idx)
            #             before_multitoken_idx = word_idx
            # else:
            #     new_example.append(word_idx)

            if before_multitoken_idx < 0:
                if word_idx != unk_id:
                    if i < len(example)-1:
                        before_multitoken_idx = word_idx
                    else:
                        new_example.append(word_idx)
                else:
                    new_example.append(word_idx)
            else:
                if before_multitoken_idx != word_idx:
                    new_example.append(before_multitoken_idx)
                    if word_idx != unk_id:
                        if i < len(example)-1:
                            before_multitoken_idx = word_idx
                        else:
                            new_example.append(word_idx)
                    else:
                        new_example.append(word_idx)

                elif i >= len(example)-1:
                    new_example.append(word_idx)


        new_batch_example_ids.append(new_example)

    new_batch_examples = []

    batch_word_sizes = []

    for example in new_batch_example_ids:

        one_example = []

        word_sizes = []

        for word_idx in example:

            split_words = seq_idx2word[word_idx].split(separator)

            one_example.extend(split_words)

            word_size = len(split_words)

            word_sizes.append(word_size)

            # one_example.append(seq_idx2word[word_idx])

        new_batch_examples.append(one_example)

        batch_word_sizes.append(word_sizes)

    return new_batch_examples,  batch_word_sizes



