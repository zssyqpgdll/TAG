import numpy as np
from copy import deepcopy

# Dataset已经废弃不用了，看 Dataset_V2
class Dataset:

    def __init__(self, dataset, pad_id, shuffle=False, use_source_seq=False):
        self.tree_data = dataset['tree_data']
        self.seq_in_data = dataset['seq_in_data']
        self.seq_out_data = dataset['seq_out_data']
        self.node_lens = dataset['node_lens']
        self.seq_lens = dataset['seq_lens']
        # self.parent_child_tables = dataset['parent_child_tables']
        # self.parent_child_types = dataset['parent_child_types']

        self.left_child_ids  = dataset['left_child_ids']
        self.right_child_ids = dataset['right_child_ids']

        self.parent_types = dataset['parent_types']
        self.left_child_types = dataset['left_child_types']
        self.right_child_types = dataset['right_child_types']
        self.tree_node_counts = dataset['tree_node_counts']
        self.gold_seqs = dataset['gold_seqs']

        self.pad_id = pad_id

        self.use_source_seq = use_source_seq

        if use_source_seq:
            self.tree_seq_in_data = dataset['tree_seq_in_data']
            self.tree_seq_out_data = dataset['tree_seq_out_data']
            self.tree_seq_lens = dataset['tree_seq_lens']

        self.shuffle = shuffle

        self.order = list(range(len(self.tree_data)))

        if shuffle:
            np.random.shuffle(self.order)

        self.current_pos = 0


    def get_batch(self, batch_size):

        start = self.current_pos

        end = min(self.current_pos + batch_size, len(self.order))

        batch_tree_data = []
        batch_seq_in_data = []
        batch_seq_out_data = []
        batch_node_lens = []
        batch_seq_lens = []
        batch_gold_seqs = []
        # batch_parent_child_tables = []
        # batch_parent_child_types = []

        batch_left_child_ids = []
        batch_right_child_ids = []

        batch_parent_types = []
        batch_left_child_types = []
        batch_right_child_types = []

        batch_tree_node_counts = []

        batch_tree_seq_in_data = []
        batch_tree_seq_out_data = []
        batch_tree_seq_lens = []

        max_node_value_len = 0

        for i in range(start, end, 1):
            td = deepcopy(self.tree_data[self.order[i]])
            batch_tree_data.append(td)

            sid = deepcopy(self.seq_in_data[self.order[i]])
            batch_seq_in_data.append(sid)

            sod = deepcopy(self.seq_out_data[self.order[i]])
            batch_seq_out_data.append(sod)

            gs = deepcopy(self.gold_seqs[self.order[i]])
            batch_gold_seqs.append(gs)

            nl = deepcopy(self.node_lens[self.order[i]])
            batch_node_lens.append(nl)
            if max(nl) >  max_node_value_len:
                max_node_value_len = max(nl)

            sl = deepcopy(self.seq_lens[self.order[i]])
            batch_seq_lens.append(sl)

            # pct = self.parent_child_tables[self.order[i]]
            # batch_parent_child_tables.append(pct)

            lcid = deepcopy(self.left_child_ids[self.order[i]])
            rcid = deepcopy(self.right_child_ids[self.order[i]])
            batch_left_child_ids.append(lcid)
            batch_right_child_ids.append(rcid)

            # pctp = self.parent_child_types[self.order[i]]
            # batch_parent_child_types.append(pctp)

            p_type = deepcopy(self.parent_types[self.order[i]])
            lc_type = deepcopy(self.left_child_types[self.order[i]])
            rc_type = deepcopy(self.right_child_types[self.order[i]])
            batch_parent_types.append(p_type)
            batch_left_child_types.append(lc_type)
            batch_right_child_types.append(rc_type)

            tnc = deepcopy(self.tree_node_counts[self.order[i]])
            # assert tnc == len(td), "tree node count != tree data nodes' number"

            batch_tree_node_counts.append(tnc)

            if self.use_source_seq:
               tsid = self.tree_seq_in_data[self.order[i]]
               batch_tree_seq_in_data.append(tsid)

               tsod = self.tree_seq_out_data[self.order[i]]
               batch_tree_seq_out_data.append(tsod)

               tsl =  self.tree_seq_lens[self.order[i]]
               batch_tree_seq_lens.append(tsl)

        self.current_pos = end

        # pad tree_data, node_lens, left_child_ids, right_child_ids, parent_types, left_child_types, right_child_types to (max node count + 1)
        # which 1 means there is at least one PAD node in the

        max_node_count = max(batch_tree_node_counts) + 1
        # print("max node count:", max_node_count)

        max_seq_len = max(batch_seq_lens)

        for i, _ in enumerate(batch_tree_data):

            # pad target sequence data: seq_in_data, seq_out_data
            batch_seq_in_data[i] = batch_seq_in_data[i] + [self.pad_id]*(max_seq_len-batch_seq_lens[i])
            batch_seq_out_data[i] = batch_seq_out_data[i] + [self.pad_id]*(max_seq_len-batch_seq_lens[i])

            for j, _ in enumerate(batch_tree_data[i]):

                batch_tree_data[i][j] = batch_tree_data[i][j] + [self.pad_id]*(max_node_value_len - len(batch_tree_data[i][j]))

            pad_num = max_node_count - batch_tree_node_counts[i]
            pad_nodes = [[self.pad_id]*max_node_value_len]*pad_num

            # pad tree data
            batch_tree_data[i].extend(pad_nodes)

            # add 0 length to node lens
            batch_node_lens[i].extend([0]*pad_num)

            real_node_num = len(batch_left_child_ids[i])

            # pad pad node index to left_child_ids
            batch_left_child_ids[i].extend([real_node_num]*pad_num)

            # pad pad node index to right_child_ids
            batch_right_child_ids[i].extend([real_node_num]*pad_num)

            # pad OTHER TYPE to parent_types
            batch_parent_types[i].extend([0]*pad_num)

            # pad OTHER TYPE bo left_child_types
            batch_left_child_types[i].extend([0]*pad_num)

            # pad OTHER TYPE to right_child_types
            batch_right_child_types[i].extend([0]*pad_num)

        # for btd in batch_tree_data:
        #     print("batch_tree_data:", np.array(btd).shape)

        batch_data = {}

        # [batch_size, max_node_count, max_seq_len]
        batch_data['batch_tree_data'] = batch_tree_data

        # [batch_size, max_seq_len]
        batch_data['batch_seq_in_data'] = batch_seq_in_data

        # [batch_size, max_seq_len]
        batch_data['batch_seq_out_data'] = batch_seq_out_data

        batch_data['batch_gold_seqs'] = batch_gold_seqs

        # [batch_size, max_node_count]
        batch_data['batch_node_lens'] = batch_node_lens

        batch_data['batch_tree_node_counts'] = batch_tree_node_counts

        # [batch_size]
        batch_data['batch_seq_lens'] = batch_seq_lens
        # batch_data['batch_parent_child_tables'] = batch_parent_child_tables
        # batch_data['batch_parent_child_types'] = batch_parent_child_types

        # [batch_size, max_node_count]
        batch_data['batch_left_child_ids'] = batch_left_child_ids

        # [batch_size, max_node_count]
        batch_data['batch_right_child_ids'] = batch_right_child_ids

        # [batch_size, max_node_count]
        batch_data['batch_parent_types'] = batch_parent_types

        # [batch_size, max_node_count]
        batch_data['batch_left_child_types'] = batch_left_child_types

        # [batch_size, max_node_count]
        batch_data['batch_right_child_types'] = batch_right_child_types

        if not self.use_source_seq:
            batch_tree_seq_in_data = None
            # batch_tree_seq_out_data = None
            batch_tree_seq_lens = None

        # [batch_size, max_seq_len] or None
        batch_data['batch_tree_seq_in_data'] = batch_tree_seq_in_data

        # [batch_size, max_seq_len] or None
        # batch_data['batch_tree_seq_out_data'] = batch_tree_seq_out_data

        # [batch_size] or None
        batch_data['batch_tree_seq_lens'] = batch_tree_seq_lens

        return batch_data

    # def pad_data(self, original_data, pad_value, max_len, use_curr_index_element_to_pad=False):
    #
    #     for i, od in enumerate(original_data):
    #
    #         if use_curr_index_element_to_pad:
    #             pad_val = od
    #         else:
    #             pad_val = pad_value
    #         original_data[i] = original_data[i] + [pad_val] * max_len
    #
    #     return original_data

    def is_end(self):
        return self.current_pos >= len(self.order)

    def reset(self):
        self.current_pos = 0

        if self.shuffle:
            np.random.shuffle(self.order)

class Dataset_V2:

    def __init__(self, dataset, pad_id, unk_id, pad_token, unk_token, max_child_num, shuffle=False, use_source_seq=False, copy=False, primitive_type_ids=None, name="trainset", copy_once=False):
        '''
        :param dataset:
        :param pad_id:
        :param max_child_num:
        :param shuffle:
        :param use_source_seq:
        :param copy: boolean, a flag to indicate whether to copy from source inputs
        :param primitive_type_ids: a list of primitive type ids, when copy is True, it should not be None
        '''

        # if copy:
        #     assert primitive_type_ids is not None, "when copy is True, primitive type ids should not be None"

        self.tree_data = dataset['tree_data']
        self.seq_in_data = dataset['seq_in_data']
        self.seq_out_data = dataset['seq_out_data']
        self.node_lens = dataset['node_lens']
        self.seq_lens = dataset['seq_lens']
        self.gold_seqs = dataset['gold_seqs']
        self.tree_data_tokens = dataset['tree_data_tokens']
        self.seq_out_tokens = dataset['seq_out_tokens']

        # self.left_child_ids  = dataset['left_child_ids']
        # self.right_child_ids = dataset['right_child_ids']
        self.child_ids = dataset['child_ids']

        self.parent_types = dataset['parent_types']
        # self.left_child_types = dataset['left_child_types']
        # self.right_child_types = dataset['right_child_types']
        self.child_types = dataset['child_types']
        self.tree_node_counts = dataset['tree_node_counts']

        self.pad_id = pad_id

        self.unk_id = unk_id

        self.pad_token = pad_token

        self.unk_token = unk_token

        self.use_source_seq = use_source_seq

        self.copy = copy

        self.copy_once = copy_once

        self.primitive_type_ids = primitive_type_ids

        self.name = name

        if use_source_seq:
            self.tree_seq_in_data = dataset['tree_seq_in_data']
            self.tree_seq_out_data = dataset['tree_seq_out_data']
            self.tree_seq_lens = dataset['tree_seq_lens']

        if copy:
            self.tree2nl_ids = dataset['tree2nl_ids']

        self.shuffle = shuffle

        self.order = list(range(len(self.tree_data)))

        if shuffle:
            np.random.shuffle(self.order)

        self.current_pos = 0

        self.max_child_num = max_child_num

    def __prepare_batch_copy_data__(self,
                                    batch_parent_types,
                                    batch_tree_node_counts,
                                    batch_tree_data,
                                    batch_seq_out_data,
                                    batch_tree_data_tokens,
                                    batch_seq_out_tokens):
        # batch_parent_types = [batch_size, max_node_count]
        # batch_tree_data = [batch_size, max_node_count, max_value_len]
        # batch_seq_out_data = [batch_size, max_time]

        batch_size = len(batch_seq_out_data)
        max_time = len(batch_seq_out_data[0])
        max_node_count = len(batch_tree_data[0])
        max_value_len = len(batch_tree_data[0][0])

        # [batch_size, max_node_count]
        encoder_mask = np.zeros_like(batch_parent_types, dtype=np.float32)

        # [batch_size, max_node_count]
        np_batch_parent_types = np.array(batch_parent_types, dtype=np.int32)

        if self.primitive_type_ids is not None and len(self.primitive_type_ids) > 0:

            for type_id in self.primitive_type_ids:

                encoder_mask = encoder_mask + (np_batch_parent_types == type_id).astype(np.float32)
        else:
            encoder_mask = np.array([(np.arange(max_node_count, dtype=np.float32) < tnc).astype(np.float32) for tnc in batch_tree_node_counts])



        # [batch_size, max_node_count, max_node_value_len]
        encoder_value_mask = np.zeros_like(batch_tree_data, dtype=np.float32)

        for i in range(len(batch_tree_data)):

            for j in range(len(batch_tree_data[i])):

                for k in range(len(batch_tree_data[i][j])):

                    if batch_tree_data[i][j][k] == self.pad_id:
                        break

                    encoder_value_mask[i, j, k] = 1.0


        if self.name == "trainset":

            # when training
            # [batch_size, max_time, max_node_count]
            tgt_encoder_mask = np.zeros(shape=[batch_size, max_time, max_node_count], dtype=np.float32)

            # [batch_size, max_time, max_node_count, max_value_len]
            # tgt_encoder_value_mask = np.zeros(shape=[batch_size, max_time, max_node_count, max_value_len], dtype=np.float32)

            # [batch_size, max_time, max_node_count, max_value_len]
            tgt_encoder_value_mask = np.tile(np.expand_dims(encoder_value_mask, axis=1), (1, max_time, 1, 1))

            for i, (seq_out_tokens, tree_data_tokens) in enumerate(zip(batch_seq_out_tokens, batch_tree_data_tokens)):
                # seq_out= [actual_time]
                # tree_data = [actual_node_count, actual_value_len]
                for j in range(len(seq_out_tokens)):

                    if seq_out_tokens[j] == self.pad_token:
                        break

                    # if seq_out[j] == self.unk_id:
                    #     continue

                    for k in range(len(tree_data_tokens)):

                        # # [max_value_len]
                        # kth_tree_data = np.array(tree_data[k], dtype=np.int32)
                        #
                        # kth_equal_flags = (kth_tree_data == seq_out[j]).astype(np.float32)
                        #
                        # # tgt_encoder_value_mask[i, j, k] = kth_equal_flags
                        #
                        # tgt_encoder_mask[i, j, k] = float(kth_equal_flags.any())

                                          # actual_value_len
                        for tree_token in tree_data_tokens[k]:

                            if tree_token == seq_out_tokens[j]:

                                if not self.copy_once:

                                    tgt_encoder_mask[i, j, k] = 1.0

                                    break
                                else:

                                    # copy once, check whether this node has ever been copied in previous time steps
                                    # [max_time]
                                    if not np.any(tgt_encoder_mask[i, :, k]):

                                        tgt_encoder_mask[i, j, k] = 1.0

                                        break

            # add primitive types contraint (if primitive_type_ids are not None)
            # [batch_size, max_time, max_node_count]
            tile_encoder_mask = np.tile(np.expand_dims(encoder_mask, axis=1), (1, max_time, 1))
            tgt_encoder_mask = tile_encoder_mask * tgt_encoder_mask

            # one_batch_should_have_primitive_num = np.sum(np.any(np.any(tgt_encoder_mask, axis=-1).astype(np.float32), axis=-1).astype(np.float32))
            # print("one_batch_should_have_primitive_num:", one_batch_should_have_primitive_num)


        else:
            # when testing, because we do not know which word is the target word
            # we just expand a max_time dimension from encoder_mask and encoder_value_mask
            # [batch_size, max_time, max_node_count]
            tgt_encoder_mask = np.tile(np.expand_dims(encoder_mask, axis=1), (1, max_time, 1))

            # [batch_size, max_time, max_node_count, max_value_len]
            tgt_encoder_value_mask = np.tile(np.expand_dims(encoder_value_mask, axis=1), (1, max_time, 1, 1))



        return encoder_mask, encoder_value_mask, tgt_encoder_mask, tgt_encoder_value_mask



    def get_batch(self, batch_size):

        start = self.current_pos

        # print("child ids length:", len(self.child_ids))

        end = min(self.current_pos + batch_size, len(self.order))

        batch_tree_data = []
        batch_seq_in_data = []
        batch_seq_out_data = []
        batch_node_lens = []
        batch_seq_lens = []
        batch_gold_seqs = []
        batch_tree_data_tokens = []
        batch_seq_out_tokens = []

        # batch_left_child_ids = []
        # batch_right_child_ids = []
        batch_child_ids = []

        batch_parent_types = []
        # batch_left_child_types = []
        # batch_right_child_types = []
        batch_child_types = []

        batch_tree_node_counts = []

        batch_tree_seq_in_data = []
        batch_tree_seq_out_data = []
        batch_tree_seq_lens = []

        batch_tree2nl_ids = []

        max_node_value_len = 0

        for i in range(start, end, 1):
            td = deepcopy(self.tree_data[self.order[i]])
            batch_tree_data.append(td)

            if self.copy:
                t2nids = deepcopy(self.tree2nl_ids[self.order[i]])
                batch_tree2nl_ids.append(t2nids)

            sid = deepcopy(self.seq_in_data[self.order[i]])
            batch_seq_in_data.append(sid)

            sod = deepcopy(self.seq_out_data[self.order[i]])
            batch_seq_out_data.append(sod)

            gs = deepcopy(self.gold_seqs[self.order[i]])
            batch_gold_seqs.append(gs)

            tdt = deepcopy(self.tree_data_tokens[self.order[i]])
            batch_tree_data_tokens.append(tdt)

            sot = deepcopy(self.seq_out_tokens[self.order[i]])
            batch_seq_out_tokens.append(sot)

            nl = deepcopy(self.node_lens[self.order[i]])
            batch_node_lens.append(nl)
            if max(nl) > max_node_value_len:
                max_node_value_len = max(nl)

            sl = deepcopy(self.seq_lens[self.order[i]])
            batch_seq_lens.append(sl)

            # pct = self.parent_child_tables[self.order[i]]
            # batch_parent_child_tables.append(pct)

            # lcid = deepcopy(self.left_child_ids[self.order[i]])
            # rcid = deepcopy(self.right_child_ids[self.order[i]])
            # batch_left_child_ids.append(lcid)
            # batch_right_child_ids.append(rcid)

            cids = deepcopy(self.child_ids[self.order[i]])
            batch_child_ids.append(cids)

            # pctp = self.parent_child_types[self.order[i]]
            # batch_parent_child_types.append(pctp)

            p_type = deepcopy(self.parent_types[self.order[i]])
            # lc_type = deepcopy(self.left_child_types[self.order[i]])
            # rc_type = deepcopy(self.right_child_types[self.order[i]])

            c_types = deepcopy(self.child_types[self.order[i]])

            batch_parent_types.append(p_type)
            # batch_left_child_types.append(lc_type)
            # batch_right_child_types.append(rc_type)
            batch_child_types.append(c_types)

            tnc = deepcopy(self.tree_node_counts[self.order[i]])
            # assert tnc == len(td), "tree node count != tree data nodes' number"

            batch_tree_node_counts.append(tnc)

            if self.use_source_seq:
               tsid = self.tree_seq_in_data[self.order[i]]
               batch_tree_seq_in_data.append(tsid)

               tsod = self.tree_seq_out_data[self.order[i]]
               batch_tree_seq_out_data.append(tsod)

               tsl =  self.tree_seq_lens[self.order[i]]
               batch_tree_seq_lens.append(tsl)

        self.current_pos = end

        assert (not self.copy) or (self.copy and max_node_value_len == 1), "in copy  mode, max_node_valen is not equal 1"
        # pad tree_data, node_lens, left_child_ids, right_child_ids, parent_types, left_child_types, right_child_types to (max node count + 1)
        # which 1 means there is at least one PAD node in the

        max_node_count = max(batch_tree_node_counts) + 1
        # print("max node count:", max_node_count)

        max_seq_len = max(batch_seq_lens)

        for i, _ in enumerate(batch_tree_data):

            # pad target sequence data: seq_in_data, seq_out_data
            batch_seq_in_data[i] = batch_seq_in_data[i] + [self.pad_id]*(max_seq_len-batch_seq_lens[i])
            batch_seq_out_data[i] = batch_seq_out_data[i] + [self.pad_id]*(max_seq_len-batch_seq_lens[i])

            for j, _ in enumerate(batch_tree_data[i]):

                batch_tree_data[i][j] = batch_tree_data[i][j] + [self.pad_id]*(max_node_value_len - len(batch_tree_data[i][j]))

                if self.name == "trainset" and self.copy:
                    batch_tree2nl_ids[i][j] = batch_tree2nl_ids[i][j] + [self.pad_id]*(max_node_value_len - len(batch_tree2nl_ids[i][j]))

            pad_num = max_node_count - batch_tree_node_counts[i]
            pad_nodes = [[self.pad_id]*max_node_value_len]*pad_num

            # pad tree data
            batch_tree_data[i].extend(pad_nodes)

            # pad tree2nl data
            if self.name == "trainset":
                copy_pad_nodes = deepcopy(pad_nodes)
            else:
                copy_pad_nodes = [[self.pad_id]*1]*pad_num

            if self.copy:
                batch_tree2nl_ids[i].extend(copy_pad_nodes)

            # add 0 length to node lens
            batch_node_lens[i].extend([0]*pad_num)

            # real_node_num = len(batch_left_child_ids[i])

            real_node_num = len(batch_child_ids[i])

            for j, _ in enumerate(batch_child_ids[i]):

                # assert len(batch_child_ids[i][j]) <= self.max_child_num, "batch_child_ids child num is larger than valid max child num"

                if len(batch_child_ids[i][j]) > self.max_child_num:
                    print("batch child ids child num is larger than valid max child num")
                    batch_child_ids[i][j] = batch_child_ids[i][j][:self.max_child_num]

                batch_child_ids[i][j] = batch_child_ids[i][j] + [real_node_num]*(self.max_child_num - len(batch_child_ids[i][j]))

            # pad pad node index to left_child_ids
            # batch_left_child_ids[i].extend([real_node_num]*pad_num)

            # pad pad node index to right_child_ids
            # batch_right_child_ids[i].extend([real_node_num]*pad_num)

            # pad pad node index to child ids
            batch_child_ids[i].extend([[real_node_num]*self.max_child_num]*pad_num)

            # pad OTHER TYPE to parent_types
            batch_parent_types[i].extend([0]*pad_num)

            # pad OTHER TYPE bo left_child_types
            # batch_left_child_types[i].extend([0]*pad_num)

            # pad OTHER TYPE to right_child_types
            # batch_right_child_types[i].extend([0]*pad_num)

            for j, _ in enumerate(batch_child_types[i]):
                # assert len(batch_child_types[i][j]) <= self.max_child_num, "batch_child_types child num is larger than valid max child num"

                if len(batch_child_types[i][j]) > self.max_child_num:
                    print("batch_child_types child num is larger than valid max child num")
                    batch_child_types[i][j] = batch_child_types[i][j][:self.max_child_num]

                batch_child_types[i][j] = batch_child_types[i][j] + [0]*(self.max_child_num - len(batch_child_types[i][j]))

            # pad OTHER TYPE to child_types
            batch_child_types[i].extend([[0]*self.max_child_num]*pad_num)

        # for btd in batch_tree_data:
        #     print("batch_tree_data:", np.array(btd).shape)

        batch_data = {}

        # [batch_size, max_node_count, max_seq_len]
        batch_data['batch_tree_data'] = batch_tree_data

        # [batch_size, max_seq_len]
        batch_data['batch_seq_in_data'] = batch_seq_in_data

        # [batch_size, max_seq_len]
        batch_data['batch_seq_out_data'] = batch_seq_out_data

        batch_data['batch_gold_seqs'] = batch_gold_seqs

        # [batch_size, max_node_count]
        batch_data['batch_node_lens'] = batch_node_lens

        batch_data['batch_tree_node_counts'] = batch_tree_node_counts

        # [batch_size]
        batch_data['batch_seq_lens'] = batch_seq_lens
        # batch_data['batch_parent_child_tables'] = batch_parent_child_tables
        # batch_data['batch_parent_child_types'] = batch_parent_child_types

        # [batch_size, max_node_count]
        # batch_data['batch_left_child_ids'] = batch_left_child_ids

        # [batch_size, max_node_count]
        # batch_data['batch_right_child_ids'] = batch_right_child_ids

        batch_data['batch_child_ids'] = batch_child_ids

        # [batch_size, max_node_count]
        batch_data['batch_parent_types'] = batch_parent_types

        # [batch_size, max_node_count]
        # batch_data['batch_left_child_types'] = batch_left_child_types

        # [batch_size, max_node_count]
        # batch_data['batch_right_child_types'] = batch_right_child_types

        batch_data['batch_child_types'] = batch_child_types

        # print("batch child ids shape:", np.array(batch_child_ids).shape)
        # print("batch child types shape:", np.array(batch_child_types).shape)

        if not self.use_source_seq:
            batch_tree_seq_in_data = None
            # batch_tree_seq_out_data = None
            batch_tree_seq_lens = None

        if self.copy:
            batch_encoder_mask, batch_encoder_value_mask, batch_tgt_encoder_mask, batch_tgt_encoder_value_mask = self.__prepare_batch_copy_data__(batch_parent_types=batch_parent_types,
                                                                                                                                                  batch_tree_node_counts=batch_tree_node_counts,
                                                                                                                                                  batch_tree_data=batch_tree_data,
                                                                                                                                                  batch_seq_out_data=batch_seq_out_data,
                                                                                                                                                  batch_tree_data_tokens=batch_tree_data_tokens,
                                                                                                                                                  batch_seq_out_tokens=batch_seq_out_tokens)
            batch_data['batch_encoder_mask'] = batch_encoder_mask
            batch_data['batch_encoder_value_mask'] = batch_encoder_value_mask
            batch_data['batch_tgt_encoder_mask'] = batch_tgt_encoder_mask
            batch_data['batch_tgt_encoder_value_mask'] = batch_tgt_encoder_value_mask

            # [batch_size, max_node_count, max_value_len] trainset
            # [batch_size, max_node_count, 1] non-trainset
            batch_data['batch_tree2nl_ids'] = batch_tree2nl_ids

        # [batch_size, max_seq_len] or None
        batch_data['batch_tree_seq_in_data'] = batch_tree_seq_in_data

        # [batch_size, max_seq_len] or None
        # batch_data['batch_tree_seq_out_data'] = batch_tree_seq_out_data

        # [batch_size] or None
        batch_data['batch_tree_seq_lens'] = batch_tree_seq_lens

        return batch_data

    # def pad_data(self, original_data, pad_value, max_len, use_curr_index_element_to_pad=False):
    #
    #     for i, od in enumerate(original_data):
    #
    #         if use_curr_index_element_to_pad:
    #             pad_val = od
    #         else:
    #             pad_val = pad_value
    #         original_data[i] = original_data[i] + [pad_val] * max_len
    #
    #     return original_data

    def is_end(self):
        return self.current_pos >= len(self.order)

    def reset(self):
        self.current_pos = 0

        if self.shuffle:
            np.random.shuffle(self.order)

