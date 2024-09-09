import tensorflow as tf
# tf.enable_eager_execution()
from treelstm.treernn import NAryTreeLSTM, NAryTreeMultiGateLSTM, NAryTreeTypeEmbedLSTM, NAryTreeMultiGateLSTM_ShareForgetGateParentTypeEmbed

import numpy as np

import tensorflow.contrib.seq2seq as seq2seq
from pointer_net import PointerNet, PointerNetData
from decoder import BasicDecoderWithCopy
from decode_utils import dynamic_decode, gumbel_softmax, gumbel_trick, classifier_reward, total_reward, classifier_reward2, bleu_reward, bleu_reward_shape, rouge_reward_shape

class TreeLSTMEncoder:

    def __init__(self, init_weight, embed_size, hidden_size, max_child_num, type_num=1, use_source_seq=False, bidirectional=False, use_type=False, layer_norm=False):

        # self.t_lstm = NAryTreeLSTM(init_weight, embed_size, hidden_size, max_child_num=max_child_num, layer_norm=layer_norm, prefix="tree_pattern_encoder")
        #
        # # if not use type , type_num is a fake number: 1
        # self.type_t_lstm = NAryTreeMultiGateLSTM(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, layer_norm=layer_norm, prefix="tree_pattern_type_encoder")

        # if not use_type:
        #     self.t_lstm = NAryTreeLSTM(init_weight, embed_size, hidden_size, max_child_num=max_child_num,
        #                                layer_norm=layer_norm, prefix="tree_pattern_encoder")
        #
        # else:
        #     self.t_lstm = NAryTreeMultiGateLSTM(init_weight, type_num, embed_size, hidden_size,
        #                                              max_child_num=max_child_num, layer_norm=layer_norm,
        #                                              prefix="tree_pattern_type_encoder")

            # self.t_lstm = NAryTreeMultiGateLSTM_ShareForgetGateParentTypeEmbed(init_weight, type_num, embed_size, hidden_size,
            #                                     max_child_num=max_child_num, layer_norm=layer_norm,
            #                                     prefix="tree_pattern_type_encoder")

            # self.t_lstm = NAryTreeTypeEmbedLSTM(init_weight,
            #                                     type_num,
            #                                     embed_size,
            #                                     hidden_size,
            #                                     max_child_num=max_child_num,
            #                                     layer_norm=layer_norm,
            #                                     prefix='tree_pattern_type_encoder')

        # when using the configuration of copy + primitive_types, use this code snippet, otherwise use the code snippet above
        self.t_lstm = NAryTreeLSTM(init_weight, embed_size, hidden_size, max_child_num=max_child_num,
                                   layer_norm=layer_norm, prefix="tree_pattern_encoder")

        self.embed_size = embed_size

        self.hidden_size = hidden_size

        self.use_source_seq = use_source_seq

        self.bidirectional = bidirectional

        self.use_type = use_type

        if use_source_seq:

            if not bidirectional:
                if layer_norm:
                    self.tree_seq_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size)
                else:
                    self.tree_seq_lstm = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, name="tree_seq_lstm_cell")
                print("create unbidirectional lstm")
            else:
                fw_num_units = int(hidden_size/2)
                bw_num_units = hidden_size - fw_num_units

                if layer_norm:
                    print("create LayerNormBasicLSTMCell")
                    self.tree_seq_fw_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=fw_num_units)
                    self.tree_seq_bw_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=bw_num_units)
                else:
                    print("create LSTMCell")
                    self.tree_seq_fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=fw_num_units, name="tree_seq_lstm_fw_cell")
                    self.tree_seq_bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=bw_num_units, name="tree_seq_lstm_bw_cell")
                print("create bidirectional lstm")


    def encode(self,
               x,
               node_states,
               child_ids,
               # left_child_ids,
               # right_child_ids,
               parent_types=None,
               child_types=None,
               # left_child_types=None,
               # right_child_types=None,
               tree_seq_in_embeds=None,
               tree_seq_lens=None):
        '''
        :param x: [batch_size, max_node_count, embed_size]
        :param node_states: [batch_size, max_node_count, 2, hidden_size]
        :param child_ids: [batch_size, max_node_count, max_child_num]
        :param parent_types: [batch_size, max_node_count]
        :param child_types: [batch_size, max_node_count, max_child_num]
        :param tree_seq_in_embeds: [batch_size, max_time, embed_size]
        :param tree_seq_lens: [batch_size]
        :return:  new_node_states list containing [2, hidden_size] state tensor (or with tree_seq_final_state a LSTMStateTuple([1, hidden_size], [1, hidden_size]))
        '''

        if self.use_source_seq:
            assert tree_seq_in_embeds is not None and tree_seq_lens is not None, "if use_source_seq is True, \
            tree_seq_in_ids and tree_seq_lens must not be None"

            # encode source sequence
            if not self.bidirectional:
                # tree_seq_outputs=[batch_size, max_time, hidden_size]
                # tree_seq_final_state=([batch_size, hidden_size], [batch_size, hidden_size])
                tree_seq_outputs, tree_seq_final_state = tf.nn.dynamic_rnn(cell=self.tree_seq_lstm,
                                                       inputs=tree_seq_in_embeds,
                                                       sequence_length=tree_seq_lens,
                                                       dtype=tf.float32,
                                                       time_major=False,
                                                       )
            else:
                # outputs = (output_fw, output_bw) = ([batch_size, max_time, cell_fw.output_size], [batch_size, max_time, cell_bw.output_size])
                # output_states =(output_state_fw, output_state_bw) = (LSTMStateTuple[batch_size, cell_fw.output_size], LSTMStateTuple[batch_size, cell_bw.output_size])
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.tree_seq_fw_lstm,
                                                cell_bw=self.tree_seq_bw_lstm,
                                                inputs=tree_seq_in_embeds,
                                                sequence_length=tree_seq_lens,
                                                dtype=tf.float32,
                                                time_major=False,
                                                )

                print("outputs:", outputs)
                print("output_states:", output_states)

                output_state_fw, output_state_bw = output_states

                # [batch_size, max_time, hidden_size]
                tree_seq_outputs = tf.concat(outputs, 2)
                print("tree_seq_outputs:", tree_seq_outputs)

                # [batch_size, hidden_size]
                # tree_seq_final_state = tf.concat(output_states, 1)
                tree_seq_final_state_h = tf.concat([output_state_fw.h, output_state_bw.h], 1)
                tree_seq_final_state_c = tf.concat([output_state_fw.c, output_state_bw.c], 1)

                tree_seq_final_state = tf.nn.rnn_cell.LSTMStateTuple(c=tree_seq_final_state_c, h=tree_seq_final_state_h)
                print("tree_seq_final_state:", tree_seq_final_state)


        t_lstm_cell = self.t_lstm
        # type_t_lstm_cell = self.type_t_lstm

        def fn(x,
               node_states,
               child_ids,
               parent_types,
               child_types,
               ):
            # x [max_node_count, embed_size]
            # node states [max_node_count, 2, hidden_size]
            # child_ids [max_node_count, max_child_num]
            # parent_types [max_node_count]
            # child_types [max_node_count, max_child_num]

            def cond(cur_node,
                     unused_x,
                     unused_node_states_ta,
                     unused_new_node_states_ta,
                     unused_child_ids,
                     unused_use_type,
                     unused_parent_types,
                     unused_child_types,
                     ):
                return cur_node >= tf.constant(0)

            def body(cur_node,
                     x,
                     node_states_ta,
                     new_node_states_ta,
                     child_ids,
                     use_type,
                     parent_types,
                     child_types,
                     ):

                # x = [max_node_count, embed_size]
                # child_ids = [max_node_count, max_child_num]
                # child_types = [max_node_count, max_child_num]
                # [max_child_num]
                one_node_child_ids = child_ids[cur_node]

                def inner_cond(index,
                               max_child_num,
                               unused_one_node_child_ids,
                               unused_cur_node,
                               unused_one_node_child_states,
                               unused_new_node_states_ta,
                               unused_node_states_ta,
                               ):
                    return index < max_child_num

                def inner_body(index,
                               unused_max_child_num,
                               one_node_child_ids,
                               cur_node,
                               one_node_child_states_ta,
                               new_node_states_ta,
                               node_states_ta,
                               ):
                    child_index = one_node_child_ids[index]

                    child_state = tf.cond(child_index > cur_node,
                                          true_fn=lambda: new_node_states_ta.read(child_index),
                                          false_fn=lambda: node_states_ta.read(child_index),
                                          )

                    one_node_child_states_ta = one_node_child_states_ta.write(index, child_state)

                    return index+1, unused_max_child_num, one_node_child_ids, cur_node, one_node_child_states_ta, new_node_states_ta, node_states_ta


                index = tf.constant(0, dtype=tf.int32)

                max_child_num = tf.shape(child_ids)[1]

                one_node_child_states_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.float32)

                inner_res = tf.while_loop(
                    cond=inner_cond,
                    body=inner_body,
                    loop_vars=[
                        index,
                        max_child_num,
                        one_node_child_ids,
                        cur_node,
                        one_node_child_states_ta,
                        new_node_states_ta,
                        node_states_ta]
                )

                cur_node = inner_res[3]
                one_node_child_states_ta = inner_res[4]
                new_node_states_ta = inner_res[5]
                node_states_ta = inner_res[6]

                # print("one_node_child_ids shape:", tf.shape(one_node_child_ids))

                # [max_child_num, 2, hidden_size]
                one_node_child_states = one_node_child_states_ta.stack()
                print(one_node_child_states)

                # print("one_node_child_states:", tf.shape(one_node_child_states))

                # # [2, hidden_size]
                # left_child_state = tf.cond(lchild_index > cur_node,
                #                            true_fn=lambda: new_node_states_ta.read(lchild_index),
                #                            false_fn=lambda: node_states_ta.read(lchild_index))


                # rchild_index = right_child_ids[cur_node]

                # # [2, hidden_size]
                # right_child_state = tf.cond(rchild_index > cur_node,
                #                            true_fn=lambda: new_node_states_ta.read(rchild_index),
                #                            false_fn=lambda: node_states_ta.read(rchild_index))

                # [1, max_child_num, hidden_size]
                child_h = tf.expand_dims(one_node_child_states[:, 0, :], axis=0)

                # print("child_h shape:", tf.shape(child_h))

                # [1, max_child_num, hidden_size]
                child_c = tf.expand_dims(one_node_child_states[:, 1, :], axis=0)

                # print("child_c shape:", tf.shape(child_c))

                # [1, embed_size]
                input = tf.expand_dims(x[cur_node, :], axis=0)

                parent_type = tf.expand_dims(parent_types[cur_node], axis=0)
                # lchild_type = tf.expand_dims(left_child_types[cur_node], axis=0)
                # rchild_type = tf.expand_dims(right_child_types[cur_node], axis=0)

                # [1, max_child_num]
                one_node_child_types = tf.expand_dims(child_types[cur_node], axis=0)

                # def true_fn(input, child_h, child_c, parent_type, child_types):
                #     return type_t_lstm_cell.forward(x=input,
                #                         child_h=child_h,
                #                         child_c=child_c,
                #                         parent_type_id=parent_type,
                #                         child_type_ids=child_types)
                #
                # def false_fn(input, child_h, child_c, parent_type, child_types):
                #     return t_lstm_cell.forward(x=input,
                #                         child_h=child_h,
                #                         child_c=child_c,
                #                         parent_type_id=parent_type,
                #                         child_type_ids=child_types,
                #                         )
                #
                # # [1, hidden_size], [1, hidden_size]
                # h_state, c_state = tf.cond(use_type,
                #         true_fn=lambda :true_fn(input, child_h, child_c, parent_type, one_node_child_types),
                #         false_fn=lambda :false_fn(input, child_h, child_c, parent_type, one_node_child_types))

                h_state, c_state = t_lstm_cell.forward(x=input, child_h=child_h, child_c=child_c, parent_type_id=parent_type, child_type_ids=one_node_child_types)

                # [2, hidden_size]
                new_node_state = tf.concat([h_state, c_state], axis=0)

                new_node_states_ta = new_node_states_ta.write(cur_node, new_node_state)

                return cur_node-1, x, node_states_ta, new_node_states_ta, child_ids, use_type, parent_types, child_types

            cur_node = tf.shape(x)[0] - 1

            use_type = tf.constant(self.use_type, dtype=tf.bool)

            node_states_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                                            clear_after_read=False).unstack(node_states)

            new_node_states_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False, element_shape=node_states[0, :, :].shape)

            res = tf.while_loop(cond=cond,
                          body=body,
                          loop_vars=[cur_node,
                                     x,
                                     node_states_ta,
                                     new_node_states_ta,
                                     child_ids,
                                     # left_child_ids,
                                     # right_child_ids,
                                     use_type,
                                     parent_types,
                                     child_types,
                                     # left_child_types,
                                     # right_child_types,
                                     ])

            new_node_states_ta = res[3]

            # [max_node_count, 2, hidden_size]
            return new_node_states_ta.stack()

        # Suppose that elems is unpacked into values, a list of tensors. The shape of the result tensor is [values.shape[0]] + fn(values[0]).shape
        # [batch_size, max_node_count, 2, hidden_size]
        encoder_states = tf.map_fn(fn=lambda inp: fn(inp[0], inp[1], inp[2], inp[3], inp[4]), elems=(x, node_states, child_ids, parent_types, child_types), dtype=tf.float32)

        if self.use_source_seq:
            return encoder_states, tree_seq_final_state

        return encoder_states


class LSTMDecoder:

    def __init__(self, hidden_size, num_layers, tgt_embeddings, tgt_vocab_size, projection_layer, mode, classifier, temperature, layer_norm=False, copy=False, tgt_with_src_in_tgt_unk_vocab_size=None):

        if copy:
            assert tgt_with_src_in_tgt_unk_vocab_size is not None, "when copy is True, tgt_with_src_in_tgt_unk_vocab_size should not be None !"

        self.tgt_embeddings = tgt_embeddings

        self.tgt_vocab_size = tgt_vocab_size

        self.tgt_with_src_in_tgt_unk_vocab_size = tgt_with_src_in_tgt_unk_vocab_size

        self.projection_layer = projection_layer

        # self.attention_layer = attention_layer

        self.hidden_size = hidden_size

        self.mode = mode

        self.copy = copy

        self.classifier = classifier

        self.temperature = temperature

        # if mode == "train":
        #     prefix = "decode_lstm_cell"
        # else:
        #     prefix = "rnn/decode_lstm_cell"

        prefix = "decode_lstm_cell"

        if num_layers > 1:

            cell_list = []
            for i in range(num_layers):
                if not layer_norm:
                    c = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, initializer=tf.contrib.layers.xavier_initializer(), name=prefix + str(i))
                else:
                    c = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size)

                cell_list.append(c)

            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells=cell_list)
        else:
            if not layer_norm:
                self.cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, initializer=tf.contrib.layers.xavier_initializer(), name=prefix)
            else:
                self.cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size)

    def built(self):
        return self.cell.built

    def __copy_from_source__(self,
                             decoder_query,
                             node_states,
                             node_value_states,
                             encoder_mask,
                             encoder_value_mask,
                             tgt_encoder_mask,
                             tgt_encoder_value_mask,
                             node_pointer_net,
                             node_value_pointer_net,
                             mode="train",
        ):
        '''
        this method is used for copy_once, different from the method __copy_from_source_with_predict_copy_once,
        it implements the copy once mask using the labels in the data preprocessing;
        while the __copy_from_source_with_predict_copy_once implement the copy once using the previous predicted looply

        :param decoder_query: [batch_size, max_time, hidden_size]
        :param node_states: [batch_size, max_node_count, hidden_size]
        :param node_value_states: [batch_size, max_node_count, max_value_len, hidden_size]
        :param encoder_mask: [batch_size, max_node_count]
        :param encoder_value_mask: [batch_size, max_node_count, max_value_len]
        :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
        :param tgt_encoder_value_mask: [batch_size, max_time, max_node_count, max_value_len]
        :param node_pointer_net:
        :param node_value_pointer_net:
        :param mode: train or other(eval, infer)
        :return:
        '''

        shape = tf.shape(node_value_states)
        batch_size = shape[0]
        max_node_count = shape[1]
        max_value_len = shape[2]
        hidden_size = shape[3]
        max_time = tf.shape(decoder_query)[1]

        # [batch_size, max_time, max_node_count]
        node_pointer_probs = node_pointer_net.forward(encoder_states=node_states, decoder_query=decoder_query, encoder_mask=encoder_mask)

        # [batch_size, 1, max_time, hidden_size]
        expand_decoder_query = tf.expand_dims(decoder_query, axis=1)

        # [batch_size, max_node_count, max_time, hidden_size]
        expand_decoder_query = tf.tile(input=expand_decoder_query, multiples=[1, max_node_count, 1, 1])

        # [batch_size*max_node_count, max_time, hidden_size]
        expand_decoder_query = tf.reshape(expand_decoder_query, shape=[-1, max_time, hidden_size])

        # [batch_size*max_node_count, max_value_len, hidden_size]
        reshape_node_value_states = tf.reshape(node_value_states, shape=[-1, max_value_len, hidden_size])

        # [batch_size*max_node_count, max_value_len]
        reshape_encoder_value_mask = tf.reshape(encoder_value_mask, shape=[-1, max_value_len])

        # [batch_size*max_node_count, max_time, max_value_len]
        node_value_pointer_probs = node_value_pointer_net.forward(encoder_states=reshape_node_value_states, decoder_query=expand_decoder_query, encoder_mask=reshape_encoder_value_mask)

        # [batch_size, max_node_count, max_time, max_value_len]
        node_value_pointer_probs = tf.reshape(node_value_pointer_probs, shape=[-1, max_node_count, max_time, max_value_len])

        # [batch_size, max_time, max_node_count, max_value_len]
        node_value_pointer_probs = tf.transpose(node_value_pointer_probs, perm=[0, 2, 1, 3])

        # [batch_size, max_time, max_node_count, max_value_len]
        # tile_node_pointer_probs = tf.tile(tf.expand_dims(node_pointer_probs, axis=3), multiples=[1, 1, 1, max_value_len])

        if mode != "train":
            # # [batch_size, max_time, max_node_count, max_value_len, max_value_len]
            # tile_infer_node_value_pointer_probs = tf.tile(tf.expand_dims(node_value_pointer_probs, axis=3), multiples=[1, 1, 1, max_value_len, 1])
            #
            # # [batch_size, max_time, max_node_count, max_value_len]
            # infer_node_value_probs = tf.reduce_sum(tile_infer_node_value_pointer_probs*tgt_encoder_value_mask, axis=4)

            # [batch_size, max_time, max_node_count, max_value_len]
            # infer_node_value_probs = node_value_pointer_probs * tgt_encoder_value_mask

            # [batch_size, max_time, max_node_count, max_value_len]
            # expand_node_pointer_probs = tf.tile(tf.expand_dims(node_pointer_probs, axis=3), multiples=[1, 1, 1, max_value_len])

            # [batch_size, max_time, max_node_count, max_value_len]
            # expand_tgt_encoder_mask = tf.tile(tf.expand_dims(tgt_encoder_mask, axis=3), multiples=[1, 1, 1, max_value_len])

            # [batch_size, max_time, max_node_count, max_value_len]
            # infer_node_value_conditioned_on_node_probs = infer_node_value_probs * expand_node_pointer_probs * expand_tgt_encoder_mask

            # [batch_size, max_time, max_node_count, 1]
            infer_node_probs = tf.expand_dims(node_pointer_probs * tgt_encoder_mask, axis=3)

            return infer_node_probs

        else:
            print_op1 = tf.print("node_pointer_probs:", node_pointer_probs)
            print_op2 = tf.print("node_value_pointer_probs:", node_value_pointer_probs)
            print_op3 = tf.print("tgt_encoder_mask:", tgt_encoder_mask)
            print_op4 = tf.print("tgt_encoder_value_mask:", tgt_encoder_value_mask)
            print_op5 = tf.print("encoder_value_mask:", encoder_value_mask)
            print_op6 = tf.print("encoder value mask has multi-token (valuel len > 1):", tf.reduce_sum(tf.cast(tf.reduce_sum(encoder_value_mask, axis=-1)>1, tf.float32)))
            print_op7 = tf.print("max_value_len:", max_value_len)

            # with tf.control_dependencies([print_op1]):
            # [batch_size, max_time, max_node_count], the reduce_sum prob should be 1
            train_node_sum_tgt_value_probs = node_pointer_probs*tf.reduce_sum(node_value_pointer_probs*tgt_encoder_value_mask, axis=-1)

            # [batch_size, max_time]
            train_output_gumbel_sample = gumbel_trick(train_node_sum_tgt_value_probs, tf.shape(train_node_sum_tgt_value_probs), self.temperature)

            # [batch_size, max_time, 1]
            expand_train_output_gumbel_sample = tf.expand_dims(train_output_gumbel_sample, axis=2)

            # [batch_size, max_time, 1]
            train_copy_output_sample_probs = tf.batch_gather(params=train_node_sum_tgt_value_probs, indices=expand_train_output_gumbel_sample)

            # [batch_size, max_time]
            train_sum_node_probs = tf.reduce_sum(train_node_sum_tgt_value_probs*tgt_encoder_mask, axis=-1)

                # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
            return train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs

    def __copy_from_source_with_predict_copy_once__(
            self,
            decoder_query,
            node_states,
            node_value_states,
            encoder_mask,
            encoder_value_mask,
            tgt_encoder_mask,
            tgt_encoder_value_mask,
            node_pointer_net,
            node_value_pointer_net,
            train_copy_vocab_probs,
            mode="train",
        ):
        '''
        used for the predict_copy_once in the training mode

        :param decoder_query: [batch_size, max_time, hidden_size]
        :param node_states: [batch_size, max_node_count, hidden_size]
        :param node_value_states: [batch_size, max_node_count, max_value_len, hidden_size]
        :param encoder_mask: [batch_size, max_node_count]
        :param encoder_value_mask: [batch_size, max_node_count, max_value_len]
        :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
        :param tgt_encoder_value_mask: [batch_size, max_time, max_node_count, max_value_len]
        :param node_pointer_net:
        :param node_value_pointer_net:
        :param train_copy_vocab_probs: [batch_size, max_time, 2]
        :param mode: train or other(eval, infer)
        :return:
        '''

        assert mode == "train", "this method is just used when the flag predict_copy_once is True in the training mode"

        shape = tf.shape(node_value_states)
        batch_size = shape[0]
        max_node_count = shape[1]
        max_value_len = shape[2]
        hidden_size = shape[3]
        max_time = tf.shape(decoder_query)[1]

        # # [batch_size, max_time, max_node_count]
        # node_pointer_probs = node_pointer_net.forward(encoder_states=node_states, decoder_query=decoder_query, encoder_mask=encoder_mask)
        #
        # # [batch_size, 1, max_time, hidden_size]
        # expand_decoder_query = tf.expand_dims(decoder_query, axis=1)
        #
        # # [batch_size, max_node_count, max_time, hidden_size]
        # expand_decoder_query = tf.tile(input=expand_decoder_query, multiples=[1, max_node_count, 1, 1])
        #
        # # [batch_size*max_node_count, max_time, hidden_size]
        # expand_decoder_query = tf.reshape(expand_decoder_query, shape=[-1, max_time, hidden_size])
        #
        # # [batch_size*max_node_count, max_value_len, hidden_size]
        # reshape_node_value_states = tf.reshape(node_value_states, shape=[-1, max_value_len, hidden_size])
        #
        # # [batch_size*max_node_count, max_value_len]
        # reshape_encoder_value_mask = tf.reshape(encoder_value_mask, shape=[-1, max_value_len])
        #
        # # [batch_size*max_node_count, max_time, max_value_len]
        # node_value_pointer_probs = node_value_pointer_net.forward(encoder_states=reshape_node_value_states, decoder_query=expand_decoder_query, encoder_mask=reshape_encoder_value_mask)
        #
        # # [batch_size, max_node_count, max_time, max_value_len]
        # node_value_pointer_probs = tf.reshape(node_value_pointer_probs, shape=[-1, max_node_count, max_time, max_value_len])
        #
        # # [batch_size, max_time, max_node_count, max_value_len]
        # node_value_pointer_probs = tf.transpose(node_value_pointer_probs, perm=[0, 2, 1, 3])


        def cond(time_step,
                 max_time,
                 unused_max_value_len,
                 unused_max_node_count,
                 unused_batch_size,
                 unused_hidden_size,
                 unused_node_states,
                 unused_decode_query,
                 unused_encoder_mask,
                 unused_node_value_states,
                 unused_encoder_value_mask,
                 unused_train_copy_vocab_probs,
                 unused_node_pointer_prob_stack,
                 unused_node_value_pointer_prob_stack,
                 unused_original_encoder_mask,
        ):
            return time_step < max_time

        def body(time_step,
                 max_time,
                 max_value_len,
                 max_node_count,
                 batch_size,
                 hidden_size,
                 node_states,
                 decoder_query,
                 encoder_mask,
                 node_value_states,
                 encoder_value_mask,
                 train_copy_vocab_probs,
                 node_pointer_prob_stack,
                 node_value_pointer_prob_stack,
                 original_encoder_mask,
                 ):
            # node_states [batch_size, max_node_count, hidden_size]
            # decoder_query [batch_size, max_time, hidden_size]
            # encoder_mask [batch_size, max_node_count]
            # node_value_states [batch_size, max_node_count, max_value_len, hidden_size]
            # encoder_value_mask [batch_size, max_node_count, max_value_len]
            # train_copy_vocab_probs [batch_size, max_time, 2]
            # original_encoder_mask [batch_size, max_node_count]

            # [batch_size, 1, hidden_size]
            cur_decoder_query = tf.expand_dims(decoder_query[:, time_step, :], axis=1)

            # [batch_size, 1, max_node_count]
            cur_node_pointer_probs = node_pointer_net.forward(encoder_states=node_states, decoder_query=cur_decoder_query,
                                                          encoder_mask=encoder_mask)

            # [batch_size, max_node_count]
            cur_node_pointer_probs = tf.squeeze(cur_node_pointer_probs, axis=1)
            node_pointer_prob_stack = node_pointer_prob_stack.write(time_step, cur_node_pointer_probs)

            # [batch_size]
            max_copy_prob_node_idx = tf.argmax(cur_node_pointer_probs, axis=-1, output_type=tf.int32)

            neg_ones = -1 * tf.ones(shape=[batch_size], dtype=tf.int32)

            # [batch_size]
            filter_idx = tf.where(train_copy_vocab_probs[:, time_step, 0] < train_copy_vocab_probs[:, time_step, 1], x=max_copy_prob_node_idx, y=neg_ones)

            # [batch_size, max_node_count]
            one_hot = tf.one_hot(
                indices=filter_idx,
                depth=max_node_count,
                dtype=tf.float32,
            )

            filter_one_hot = original_encoder_mask * one_hot

            # [batch_size, max_node_count]
            # encoder_mask = encoder_mask - filter_one_hot

            def fn(encoder_mask, filter_one_hot):
                # encoder_mask [max_node_count]
                # filter_one_hot [max_node_count]
                return tf.logical_or(tf.reduce_all((encoder_mask - filter_one_hot) >= 0.0),
                                     tf.reduce_all(tf.equal(encoder_mask, 0.0)))

            # [batch_size]
            check_result = tf.map_fn(fn=lambda x: fn(x[0], x[1]), elems=(encoder_mask, filter_one_hot), dtype=tf.bool)

            assert_op = tf.Assert(tf.reduce_all(check_result), [
                'encoder mask - filter one hot some elements are < 0, they must be >= 0 for validation'])

            # assert_op = tf.Assert(tf.reduce_all((encoder_mask-filter_one_hot) >= 0.0), ['encoder mask - filter one hot some elements are < 0, they must be >= 0 for validation'])
            # print_op = tf.print("after mask:", encoder_mask-filter_one_hot, summarize=-1)
            # print_op2 = tf.print("min mask value:", tf.reduce_min(encoder_mask-filter_one_hot))

            with tf.control_dependencies([assert_op]):

                # [batch_size, max_node_count]
                encoder_mask = tf.where((encoder_mask-filter_one_hot) >= 0.0, x=encoder_mask-filter_one_hot, y=encoder_mask)

            # [batch_size, 1, 1, hidden_size]
            expand_cur_decoder_query = tf.expand_dims(cur_decoder_query, axis=1)

            # [batch_size, max_node_count, 1, hidden_size]
            expand_cur_decoder_query = tf.tile(input=expand_cur_decoder_query, multiples=[1, max_node_count, 1, 1])

            # [batch_size*max_node_count, 1, hidden_size]
            expand_cur_decoder_query = tf.reshape(expand_cur_decoder_query, shape=[-1, 1, hidden_size])

            # [batch_size*max_node_count, max_value_len, hidden_size]
            reshape_node_value_states = tf.reshape(node_value_states, shape=[-1, max_value_len, hidden_size])

            # [batch_size*max_node_count, max_value_len]
            reshape_encoder_value_mask = tf.reshape(encoder_value_mask, shape=[-1, max_value_len])

            # [batch_size*max_node_count, 1, max_value_len]
            cur_node_value_pointer_probs = node_value_pointer_net.forward(encoder_states=reshape_node_value_states,
                                                                      decoder_query=expand_cur_decoder_query,
                                                                      encoder_mask=reshape_encoder_value_mask)

            # [batch_size, max_node_count, 1, max_value_len]
            cur_node_value_pointer_probs = tf.reshape(cur_node_value_pointer_probs,
                                                  shape=[-1, max_node_count, 1, max_value_len])

            # [batch_size, max_node_count, max_value_len]
            cur_node_value_pointer_probs = tf.squeeze(cur_node_value_pointer_probs, axis=2)

            node_value_pointer_prob_stack = node_value_pointer_prob_stack.write(time_step, cur_node_value_pointer_probs)


            return time_step+1, \
                   max_time, \
                   max_value_len, \
                   max_node_count, \
                   batch_size, \
                   hidden_size, \
                   node_states, \
                   decoder_query, \
                   encoder_mask, \
                   node_value_states, \
                   encoder_value_mask, \
                   train_copy_vocab_probs, \
                   node_pointer_prob_stack, \
                   node_value_pointer_prob_stack, \
                   original_encoder_mask

        initial_time_step = tf.constant(0, dtype=tf.int32)

        initial_encoder_mask = tf.identity(encoder_mask)

        original_encoder_mask = tf.identity(encoder_mask)

        node_pointer_prob_stack = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        node_value_pointer_prob_stack = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        copy_res = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                initial_time_step,
                max_time,
                max_value_len,
                max_node_count,
                batch_size,
                hidden_size,
                node_states,
                decoder_query,
                initial_encoder_mask,
                node_value_states,
                encoder_value_mask,
                train_copy_vocab_probs,
                node_pointer_prob_stack,
                node_value_pointer_prob_stack,
                original_encoder_mask,
            ],

        )

        # [max_time, batch_size, max_node_count]
        node_pointer_probs = copy_res[12].stack()

        # [batch_size, max_time, max_node_count]
        node_pointer_probs = tf.transpose(node_pointer_probs, perm=[1, 0, 2])

        # [max_time, batch_size, max_node_count, max_value_len]
        node_value_pointer_probs = copy_res[13].stack()

        # [batch_size, max_time, max_node_count, max_value_len]
        node_value_pointer_probs = tf.transpose(node_value_pointer_probs, perm=[1, 0, 2, 3])

        # if mode != "train":
        #
        #     # [batch_size, max_time, max_node_count, 1]
        #     infer_node_probs = tf.expand_dims(node_pointer_probs * tgt_encoder_mask, axis=3)
        #
        #     return infer_node_probs
        #
        # else:

        # with tf.control_dependencies([print_op1]):
        # [batch_size, max_time, max_node_count]
        train_node_sum_tgt_value_probs = node_pointer_probs*tf.reduce_sum(node_value_pointer_probs*tgt_encoder_value_mask, axis=-1)

        # [batch_size, max_time]
        train_output_gumbel_sample = gumbel_trick(train_node_sum_tgt_value_probs, tf.shape(train_node_sum_tgt_value_probs), self.temperature)

        # [batch_size, max_time, 1]
        expand_train_output_gumbel_sample = tf.expand_dims(train_output_gumbel_sample, axis=2)

        # [batch_size, max_time, 1]
        train_copy_output_sample_probs = tf.batch_gather(params=train_node_sum_tgt_value_probs, indices=expand_train_output_gumbel_sample)

        # [batch_size, max_time]
        train_sum_node_probs = tf.reduce_sum(train_node_sum_tgt_value_probs*tgt_encoder_mask, axis=-1)

        # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
        return train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs


    def __copy_from_source_with_copy_decay__(
            self,
            decoder_query,
            node_states,
            node_value_states,
            encoder_mask,
            encoder_value_mask,
            tgt_encoder_mask,
            tgt_encoder_value_mask,
            node_pointer_net,
            node_value_pointer_net,
            train_copy_vocab_probs,
            tree2nl_ids,
            copy_decay_keep_prob,
            mode="train",
        ):
        '''
        used for the predict_copy_once in the training mode

        :param decoder_query: [batch_size, max_time, hidden_size]
        :param node_states: [batch_size, max_node_count, hidden_size]
        :param node_value_states: [batch_size, max_node_count, max_value_len, hidden_size]
        :param encoder_mask: [batch_size, max_node_count]
        :param encoder_value_mask: [batch_size, max_node_count, max_value_len]
        :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
        :param tgt_encoder_value_mask: [batch_size, max_time, max_node_count, max_value_len]
        :param node_pointer_net:
        :param node_value_pointer_net:
        :param train_copy_vocab_probs: [batch_size, max_time, 2]
        :param tree2nl_ids: [batch_size, max_node_count, max_value_len(1)]
        :param mode: train or other(eval, infer)
        :return:
        '''

        assert mode == "train", "this method is just used when the flag predict_copy_once is True in the training mode"

        shape = tf.shape(node_value_states)
        batch_size = shape[0]
        max_node_count = shape[1]
        max_value_len = shape[2]
        hidden_size = shape[3]
        max_time = tf.shape(decoder_query)[1]


        def cond(time_step,
                 max_time,
                 unused_max_value_len,
                 unused_max_node_count,
                 unused_batch_size,
                 unused_hidden_size,
                 unused_node_states,
                 unused_decode_query,
                 unused_encoder_mask,
                 unused_node_value_states,
                 unused_encoder_value_mask,
                 unused_train_copy_vocab_probs,
                 unused_node_pointer_prob_stack,
                 unused_node_value_pointer_prob_stack,
                 unused_original_encoder_mask,
                 unused_tree2nl_ids,
                 unused_copy_decay_rates,
                 unused_tgt_vocab_size,
                 unused_copy_decay_keep_prob,
                 unused_copy_time_record,
        ):
            return time_step < max_time

        def body(time_step,
                 max_time,
                 max_value_len,
                 max_node_count,
                 batch_size,
                 hidden_size,
                 node_states,
                 decoder_query,
                 encoder_mask,
                 node_value_states,
                 encoder_value_mask,
                 train_copy_vocab_probs,
                 node_pointer_prob_stack,
                 node_value_pointer_prob_stack,
                 original_encoder_mask,
                 tree2nl_ids,
                 copy_decay_rates,
                 tgt_vocab_size,
                 copy_decay_keep_prob,
                 copy_time_record,
                 ):
            # node_states [batch_size, max_node_count, hidden_size]
            # decoder_query [batch_size, max_time, hidden_size]
            # encoder_mask [batch_size, max_node_count]
            # node_value_states [batch_size, max_node_count, max_value_len, hidden_size]
            # encoder_value_mask [batch_size, max_node_count, max_value_len]
            # train_copy_vocab_probs [batch_size, max_time, 2]
            # original_encoder_mask [batch_size, max_node_count]
            # tree2nl_ids [batch_size, max_node_count]
            # copy_decay_rates [batch_size, max_node_count]
            # copy_time_record [batch_size, max_node_count]

            # [batch_size, 1, hidden_size]
            cur_decoder_query = tf.expand_dims(decoder_query[:, time_step, :], axis=1)

            # [batch_size, 1, max_node_count]
            cur_node_pointer_probs = node_pointer_net.forward(encoder_states=node_states, decoder_query=cur_decoder_query,
                                                          encoder_mask=encoder_mask)

            # [batch_size, max_node_count]
            cur_node_pointer_probs = tf.squeeze(cur_node_pointer_probs, axis=1) * (1.0 - copy_decay_rates)

            node_pointer_prob_stack = node_pointer_prob_stack.write(time_step, cur_node_pointer_probs)

            # [batch_size]
            max_copy_prob_node_idx = tf.argmax(cur_node_pointer_probs, axis=-1, output_type=tf.int32)

            # [batch_size]
            max_copy_prob_node_idx = tf.where(train_copy_vocab_probs[:, time_step, 0] < train_copy_vocab_probs[:, time_step, 1], x=max_copy_prob_node_idx, y=-1*tf.ones(shape=[batch_size], dtype=tf.int32))

            # [batch_size, max_node_count]
            one_hot = tf.one_hot(indices=max_copy_prob_node_idx, depth=max_node_count, dtype=tf.int32)

            # [batch_size, max_node_count]
            copy_time_record = copy_time_record + one_hot

            # [batch_size, max_node_count]
            copy_decay_rates = copy_decay_rates*copy_decay_keep_prob

            # [batch_size, max_node_count]
            copy_decay_rates = tf.where(one_hot > 0, x=copy_decay_keep_prob*tf.ones_like(copy_decay_rates, dtype=tf.float32), y=copy_decay_rates)


            # # [batch_size, 1]
            # expand_max_copy_prob_node_idx = tf.expand_dims(max_copy_prob_node_idx, axis=1)
            #
            # # [batch_size, 1]
            # gather_tree2nl_ids = tf.batch_gather(params=tree2nl_ids, indices=expand_max_copy_prob_node_idx)
            #
            # # [batch_size, tgt_vocab_size]
            # tree2nl_id_ont_hot = tf.one_hot(indices=tf.squeeze(gather_tree2nl_ids, axis=1), depth=tgt_vocab_size, dtype=tf.float32)
            #
            # # [batch_size, tgt_vocab_size]
            # updated_copy_record_pos = tree2nl_id_ont_hot
            #
            # # [batch_size, tgt_vocab_size]
            # non_updated_copy_record_pos = 1.0 - tree2nl_id_ont_hot
            #
            # copy_distance = updated_copy_record_pos * time_step - updated_copy_record_pos * copy_time_step_record
            #
            # # [batch_size, tgt_vocab_size]
            # copy_decay = tf.where(copy_time_step_record < 0, x=tf.zeros_like(copy_time_step_record, dtype=tf.float32), y=tf.pow(copy_decay_rate, copy_distance))
            #
            # # [batch_size, tgt_vocab_size]
            # copy_time_step_record = non_updated_copy_record_pos * copy_time_step_record + updated_copy_record_pos * time_step


            # [batch_size, 1, 1, hidden_size]
            expand_cur_decoder_query = tf.expand_dims(cur_decoder_query, axis=1)

            # [batch_size, max_node_count, 1, hidden_size]
            expand_cur_decoder_query = tf.tile(input=expand_cur_decoder_query, multiples=[1, max_node_count, 1, 1])

            # [batch_size*max_node_count, 1, hidden_size]
            expand_cur_decoder_query = tf.reshape(expand_cur_decoder_query, shape=[-1, 1, hidden_size])

            # [batch_size*max_node_count, max_value_len, hidden_size]
            reshape_node_value_states = tf.reshape(node_value_states, shape=[-1, max_value_len, hidden_size])

            # [batch_size*max_node_count, max_value_len]
            reshape_encoder_value_mask = tf.reshape(encoder_value_mask, shape=[-1, max_value_len])

            # [batch_size*max_node_count, 1, max_value_len]
            cur_node_value_pointer_probs = node_value_pointer_net.forward(encoder_states=reshape_node_value_states,
                                                                      decoder_query=expand_cur_decoder_query,
                                                                      encoder_mask=reshape_encoder_value_mask)

            # [batch_size, max_node_count, 1, max_value_len]
            cur_node_value_pointer_probs = tf.reshape(cur_node_value_pointer_probs,
                                                  shape=[-1, max_node_count, 1, max_value_len])

            # [batch_size, max_node_count, max_value_len]
            cur_node_value_pointer_probs = tf.squeeze(cur_node_value_pointer_probs, axis=2)

            node_value_pointer_prob_stack = node_value_pointer_prob_stack.write(time_step, cur_node_value_pointer_probs)


            return time_step+1, \
                   max_time, \
                   max_value_len, \
                   max_node_count, \
                   batch_size, \
                   hidden_size, \
                   node_states, \
                   decoder_query, \
                   encoder_mask, \
                   node_value_states, \
                   encoder_value_mask, \
                   train_copy_vocab_probs, \
                   node_pointer_prob_stack, \
                   node_value_pointer_prob_stack, \
                   original_encoder_mask, \
                   tree2nl_ids, \
                   copy_decay_rates, \
                   tgt_vocab_size, \
                   copy_decay_keep_prob, \
                   copy_time_record

        initial_time_step = tf.constant(0, dtype=tf.int32)

        initial_encoder_mask = tf.identity(encoder_mask)

        initial_copy_decay_rates = tf.zeros(shape=[batch_size, max_node_count], dtype=tf.float32)

        initial_copy_time_record = tf.zeros(shape=[batch_size, max_node_count], dtype=tf.int32)

        original_encoder_mask = tf.identity(encoder_mask)

        # [batch_size, max_node_count]
        initial_tree2nl_ids = tf.squeeze(tree2nl_ids, axis=-1)

        node_pointer_prob_stack = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        node_value_pointer_prob_stack = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        copy_res = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                initial_time_step,
                max_time,
                max_value_len,
                max_node_count,
                batch_size,
                hidden_size,
                node_states,
                decoder_query,
                initial_encoder_mask,
                node_value_states,
                encoder_value_mask,
                train_copy_vocab_probs,
                node_pointer_prob_stack,
                node_value_pointer_prob_stack,
                original_encoder_mask,
                initial_tree2nl_ids,
                initial_copy_decay_rates,
                self.tgt_vocab_size,
                copy_decay_keep_prob,
                initial_copy_time_record,
            ],

        )

        # [max_time, batch_size, max_node_count]
        node_pointer_probs = copy_res[12].stack()

        # [batch_size, max_time, max_node_count]
        node_pointer_probs = tf.transpose(node_pointer_probs, perm=[1, 0, 2])

        # [max_time, batch_size, max_node_count, max_value_len]
        node_value_pointer_probs = copy_res[13].stack()

        # [batch_size, max_time, max_node_count, max_value_len]
        node_value_pointer_probs = tf.transpose(node_value_pointer_probs, perm=[1, 0, 2, 3])

        # with tf.control_dependencies([print_op1]):
        # [batch_size, max_time, max_node_count]
        train_node_sum_tgt_value_probs = node_pointer_probs*tf.reduce_sum(node_value_pointer_probs*tgt_encoder_value_mask, axis=-1)

        # [batch_size, max_time]
        train_output_gumbel_sample = gumbel_trick(train_node_sum_tgt_value_probs, tf.shape(train_node_sum_tgt_value_probs), self.temperature)

        # [batch_size, max_time, 1]
        expand_train_output_gumbel_sample = tf.expand_dims(train_output_gumbel_sample, axis=2)

        # [batch_size, max_time, 1]
        train_copy_output_sample_probs = tf.batch_gather(params=train_node_sum_tgt_value_probs, indices=expand_train_output_gumbel_sample)

        # [batch_size, max_time]
        train_sum_node_probs = tf.reduce_sum(train_node_sum_tgt_value_probs*tgt_encoder_mask, axis=-1)

        # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
        return train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs


    def __copy_from_source_one_step__(self,
                             decoder_query,
                             node_states,
                             node_value_states,
                             encoder_mask,
                             encoder_value_mask,
                             tgt_encoder_mask,
                             tgt_encoder_value_mask,
                             node_pointer_net,
                             node_value_pointer_net,
                             decoder_time_step,
        ):
        '''
        calculate one-step copy probability for decoder (when inferencing or evaluating)
        :param decoder_query: [batch_size, hidden_size]
        :param node_states: [batch_size, max_node_count, hidden_size]
        :param node_value_states: [batch_size, max_node_count, max_value_len, hidden_size]
        :param encoder_mask: [batch_size, max_node_count]
        :param encoder_value_mask: [batch_size, max_node_count, max_value_len]
        :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
        :param tgt_encoder_value_mask: [batch_size, max_time, max_node_count, max_value_len]
        :param node_pointer_net:
        :param node_value_pointer_net:
        :param decoder_time_step: current decoder time step
        :return:
        '''

        # [batch_size, 1, hidden_size]
        cur_step_decoder_query = tf.expand_dims(decoder_query, axis=1)

        # in inference, because we do not know which is target, so just use the first time step mask
        # use the current time step may be dangerous to be out of thd array bound, because we use the target sequence max_time
        # may be less than the inference max time
        # [batch_size, 1, max_node_count]
        cur_step_tgt_encoder_mask = tf.expand_dims(tgt_encoder_mask[:, 0, :], axis=1)
        # cur_step_tgt_encoder_mask = tf.expand_dims(tgt_encoder_mask[:, decoder_time_step, :], axis=1)

        # [batch_size, 1, max_node_count, max_value_len]
        # unused, just as an unused parameter
        cur_step_infer_tgt_encoder_value_mask = tf.expand_dims(tgt_encoder_value_mask[:, 0, :, :], axis=1)
        # unused, just as an unused parameter
        # cur_step_infer_tgt_encoder_value_mask = tf.expand_dims(tgt_encoder_value_mask[:, decoder_time_step, :, :], axis=1)

        # [batch_size, 1, max_node_count, max_value_len]
        cur_step_infer_node_value_probs = self.__copy_from_source__(
                                                             decoder_query=cur_step_decoder_query,
                                                             node_states=node_states,
                                                             node_value_states=node_value_states,
                                                             encoder_mask=encoder_mask,
                                                             encoder_value_mask=encoder_value_mask,
                                                             tgt_encoder_mask=cur_step_tgt_encoder_mask,
                                                             tgt_encoder_value_mask=cur_step_infer_tgt_encoder_value_mask,
                                                             node_pointer_net=node_pointer_net,
                                                             node_value_pointer_net=node_value_pointer_net,
                                                             mode="infer",)

        # [batch_size, max_node_count, max_value_len]
        cur_step_infer_node_value_probs = tf.squeeze(cur_step_infer_node_value_probs, axis=1)

        # [batch_size, max_node_count, max_value_len]
        return cur_step_infer_node_value_probs


    def decode(self,
               train_inputs,
               encoder_state,
               sos_id,
               eos_id,
               unk_id,
               batch_parent_types,
               memory=None,
               tree_node_counts=None,
               decoder_train_lengths=None,
               max_iterations=None,
               use_attention=True,
               pointer_net_data=None,
               copy_once=False,
               predict_copy_once=False,
               copy_decay=False,
        ):
        '''
        use once to decode a batch target sequecne.

        :param train_inputs: training input ids [batch_size, max-time]
        :param encoder_state: LSTMStateTuple(h, c), h=[batch_size, hidden_size], c=[batch_size, hidden_size]
        :param batch_parent_types: [batch_size, max_node_count]
        :param decoder_train_lengths: a batch sequence actually length, shape=[batch_size]
        :param max_iterations: used when feed_prev is True, if feed_prev is False, it will not be used
        :param use_attention: bool, if True , applying the attention mechanism to the decoder
        :param memory: encoder all time-steps output, shape=[batch_size, max_node_count, hidden_size]
        :param pointer_net_data: PointerNetData

        :return:sample_id
                decoder_infer=[batch_size, max-time, tgt_vocab_size]
                decoder_train=[batch_size, max-time, tgt_vocab_size] or no_op
                train_sample_id
        '''

        if self.copy:
            assert pointer_net_data is not None, \
                "when copy is True, pointer_net_data and node_value_states must not be None"

        input_shape = tf.shape(train_inputs)
        batch_size = input_shape[0]
        max_time = input_shape[1]

        if use_attention:
            assert memory is not None and tree_node_counts is not None, "when use_attention is True, memory and tree_node_counts should not be None"
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.hidden_size, memory,
                                                                    memory_sequence_length=tree_node_counts)
            self.cell = seq2seq.AttentionWrapper(self.cell, attention_mechanism,
                                                 attention_layer_size=self.hidden_size,
                                                 alignment_history=True)
            decoder_initial_state = self.cell.zero_state(batch_size, tf.float32).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        # print(sos_id)
        # print(batch_size)
        decoder_infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.tgt_embeddings,
                                                                        tf.ones(shape=[batch_size], dtype=tf.int32),
                                                                        eos_id)

        if not self.copy:

            decoder_infer = tf.contrib.seq2seq.BasicDecoder(self.cell, decoder_infer_helper,
                                                            decoder_initial_state,
                                                            output_layer=self.projection_layer)

            decoder_outputs_infer, decoder_states_infer, decoder_seq_len_infer = seq2seq.dynamic_decode(decoder_infer,
                                                                                                        maximum_iterations=max_iterations)
        else:

            # [batch_size, max_node_count, hidden_size]
            # node_states = memory

            # [batch_size, max_node_count, hidden_size]
            node_states = pointer_net_data.node_states

            decoder_infer = BasicDecoderWithCopy(cell=self.cell,
                 helper=decoder_infer_helper,
                 initial_state=decoder_initial_state,
                 output_layer=self.projection_layer,
                 copy_score_fn=self.__copy_from_source_one_step__,
                 node_states=node_states,
                 node_value_states=pointer_net_data.node_value_states,
                 encoder_mask=pointer_net_data.encoder_mask,
                 encoder_value_mask=pointer_net_data.encoder_value_mask,
                 tgt_encoder_mask=pointer_net_data.tgt_encoder_mask,
                 tgt_encoder_value_mask=pointer_net_data.tgt_encoder_value_mask,
                 node_pointer_net=pointer_net_data.node_pointer_net,
                 node_value_pointer_net=pointer_net_data.node_value_pointer_net,
                 copy_vocab_probs_layer=pointer_net_data.copy_vocab_probs_layer,
                 tree2nl_ids=pointer_net_data.tree2nl_ids,
                 tgt_vocab_size=self.tgt_vocab_size,
                 tgt_with_src_in_tgt_unk_vocab_size=self.tgt_with_src_in_tgt_unk_vocab_size,
                 unk_id=unk_id,
                 classifier=self.classifier,
                 temperature=self.temperature,
                 copy_decay_keep_prob=pointer_net_data.copy_decay_keep_prob,
                 copy_once=copy_once,
                 predict_copy_once=predict_copy_once,
                 copy_decay=copy_decay,
            )

            decoder_outputs_infer, decoder_states_infer, decoder_seq_len_infer = dynamic_decode(decoder_infer,
                                                                                                maximum_iterations=max_iterations)

            # # [batch_size, max_time, hidden_size]
            # infer_attention_vectors = decoder_outputs_infer.rnn_output
            #
            # # [batch_size, max_time, 2]
            # infer_copy_vocab_probs = pointer_net_data.copy_vocab_probs_layer(infer_attention_vectors)
            #
            # # [batch_size, max_node_count, hidden_size]
            # node_states = memory
            #
            # # [batch_size, max_time]
            # infer_sum_node_probs = self.__copy_from_source__(decoder_query=infer_attention_vectors,
            #                                                  node_states=node_states,
            #                                                  node_value_states=node_value_states,
            #                                                  encoder_mask=pointer_net_data.encoder_mask,
            #                                                  encoder_value_mask=pointer_net_data.encoder_value_mask,
            #                                                  tgt_encoder_mask=pointer_net_data.tgt_encoder_mask,
            #                                                  train_tgt_encoder_value_mask=pointer_net_data.train_tgt_encoder_value_mask,
            #                                                  infer_tgt_encoder_value_mask=pointer_net_data.infer_tgt_encoder_value_mask,
            #                                                  node_pointer_net=pointer_net_data.node_pointer_net,
            #                                                  node_value_pointer_net=pointer_net_data.node_value_pointer_net)

        # decoder_outputs_infer, decoder_states_infer, decoder_seq_len_infer = seq2seq.dynamic_decode(decoder_infer,
        #                                                                                             maximum_iterations=max_iterations)

        decoder_infer_attn_weights = decoder_states_infer.alignment_history.stack()

        decoder_logits_infer = decoder_outputs_infer.rnn_output

        # [batch_size, max_time]
        sample_id = decoder_outputs_infer.sample_id

        one_batch_infer_copy_time_steps = tf.reduce_sum(tf.cast(tf.greater_equal(sample_id, self.tgt_vocab_size), tf.float32))

        if self.mode == "train":

            # [batchs_size, max-len, embed_size]
            decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.tgt_embeddings, train_inputs)

            decoder_train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_train_inputs_embedded,
                                                                     decoder_train_lengths)
            decoder_train = tf.contrib.seq2seq.BasicDecoder(self.cell, decoder_train_helper,
                                                 decoder_initial_state)
            decoder_outputs_train, decoder_states_train, decoder_seq_len_train = tf.contrib.seq2seq.dynamic_decode(decoder_train)

            # ------------training attention weights info-------------------
            # [max_time, batch_size, alignment_size(memory's max_time)]
            decoder_train_attn_weights = decoder_states_train.alignment_history.stack()

            # [batch_size, max_time, alignment_size]
            decoder_train_attn_weights = tf.transpose(decoder_train_attn_weights, perm=[1, 0, 2])

            # [batch_size, max_time, 1]
            attn_weights_max_idx = tf.expand_dims(tf.argmax(decoder_train_attn_weights, axis=-1, output_type=tf.int32), axis=2)

            # [batch_size, max_time, max_node_count]
            batch_parent_types = tf.tile(tf.expand_dims(batch_parent_types, axis=1), multiples=[1, tf.shape(attn_weights_max_idx)[1], 1])

            # [batch_size, max_time]
            max_attn_types = tf.squeeze(tf.batch_gather(batch_parent_types, attn_weights_max_idx), axis=2)

            one_batch_attention_to_primitive_num = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.cast((tf.equal(max_attn_types, 2)), tf.float32) + tf.cast((tf.equal(max_attn_types, 4)), tf.float32), axis=1) > 0, tf.float32))
            # ---------------------------------------------------------------

            print_op1 = tf.print("train_attn_weights-shape:", tf.shape(decoder_train_attn_weights))
            print_op2 = tf.print("train_attn_weights:", decoder_train_attn_weights)
            print_op3 = tf.print("one_batch_attention_to_primitive_num:", one_batch_attention_to_primitive_num)

            # with tf.control_dependencies([print_op3]):

            decoder_logits_train = self.projection_layer(decoder_outputs_train.rnn_output)

            decoder_logits_train = tf.reshape(decoder_logits_train, [batch_size, -1, self.tgt_vocab_size])

            if self.copy:

                # [batch_size, max_time, hidden_size]
                train_attention_vectors = decoder_outputs_train.rnn_output

                # [batch_size, max_time, 2]
                train_copy_vocab_probs = pointer_net_data.copy_vocab_probs_layer(train_attention_vectors)

                if self.classifier == "gumbel" or self.classifier == "rl":
                    # # in gumbel softmax
                    # # traim_copy_vocab_probs is logit
                    # # [batch_size, max_time, 2]
                    # train_gumbel_sample = gumbel_softmax(train_copy_vocab_probs, tf.shape(train_copy_vocab_probs), temperature=self.temperature)
                    #
                    # # add softmax activation into train_copy_vocab_probs
                    # train_copy_vocab_probs = tf.nn.softmax(train_copy_vocab_probs)

                    # [batch_size, max_time]
                    train_classifier_gumbel_sample = gumbel_trick(train_copy_vocab_probs, tf.shape(train_copy_vocab_probs), temperature=self.temperature)
                else:
                    # in hard & soft classifier
                    # train_copy_vocab_probs is probs through softmax activation function
                    train_classifier_gumbel_sample = tf.no_op()

                # [batch_size, max_node_count, hidden_size]
                # node_states = memory

                # [batch_size, max_node_count, hidden_size]
                node_states = pointer_net_data.node_states

                if predict_copy_once:
                    # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
                    train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs = self.__copy_from_source_with_predict_copy_once__(
                        decoder_query=train_attention_vectors,
                        node_states=node_states,
                        node_value_states=pointer_net_data.node_value_states,
                        encoder_mask=pointer_net_data.encoder_mask,
                        encoder_value_mask=pointer_net_data.encoder_value_mask,
                        tgt_encoder_mask=pointer_net_data.tgt_encoder_mask,
                        tgt_encoder_value_mask=pointer_net_data.tgt_encoder_value_mask,
                        node_pointer_net=pointer_net_data.node_pointer_net,
                        node_value_pointer_net=pointer_net_data.node_value_pointer_net,
                        train_copy_vocab_probs=train_copy_vocab_probs,
                        mode="train",
                    )
                elif copy_decay:
                    # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
                    train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs = self.__copy_from_source_with_copy_decay__(
                        decoder_query=train_attention_vectors,
                        node_states=node_states,
                        node_value_states=pointer_net_data.node_value_states,
                        encoder_mask=pointer_net_data.encoder_mask,
                        encoder_value_mask=pointer_net_data.encoder_value_mask,
                        tgt_encoder_mask=pointer_net_data.tgt_encoder_mask,
                        tgt_encoder_value_mask=pointer_net_data.tgt_encoder_value_mask,
                        node_pointer_net=pointer_net_data.node_pointer_net,
                        node_value_pointer_net=pointer_net_data.node_value_pointer_net,
                        train_copy_vocab_probs=train_copy_vocab_probs,
                        tree2nl_ids=pointer_net_data.tree2nl_ids,
                        copy_decay_keep_prob=pointer_net_data.copy_decay_keep_prob,
                        mode="train",
                    )
                else:
                    # copy once
                    # [batch_size, max_time], [batch_size, max_time], [batch_size, max_time, 1]
                    train_sum_node_probs, train_output_gumbel_sample, train_copy_output_sample_probs = self.__copy_from_source__(
                                                                 decoder_query=train_attention_vectors,
                                                                 node_states=node_states,
                                                                 node_value_states=pointer_net_data.node_value_states,
                                                                 encoder_mask=pointer_net_data.encoder_mask,
                                                                 encoder_value_mask=pointer_net_data.encoder_value_mask,
                                                                 tgt_encoder_mask=pointer_net_data.tgt_encoder_mask,
                                                                 tgt_encoder_value_mask=pointer_net_data.tgt_encoder_value_mask,
                                                                 node_pointer_net=pointer_net_data.node_pointer_net,
                                                                 node_value_pointer_net=pointer_net_data.node_value_pointer_net,
                                                                 mode="train")

                if self.classifier == "rl":
                    # pointer_net_data.tree2nl_ids = [batch_size, max_node_count, max_value_len(1)]
                    # train_node_sum_tgt_value_probs = [batch_size, max_time, max_node_count]
                    assert_op = tf.Assert(tf.equal(tf.shape(pointer_net_data.tree2nl_ids)[2], 1), ['tree2nl_ids dim 3 is not equal 1'])

                    with tf.control_dependencies([assert_op]):
                        # here max_value_len dim's size should be 1
                        # [batch_size, max_node_count]
                        squeeze_tree2nl_ids = tf.squeeze(pointer_net_data.tree2nl_ids, axis=2)

                        # [batch_size, max_time, max_node_count]
                        tile_tree2nl_ids = tf.tile(tf.expand_dims(squeeze_tree2nl_ids, axis=1), multiples=[1, max_time, 1])

                        # [batch_size, max_time, 1]
                        expand_train_output_gumbel_sample = tf.expand_dims(train_output_gumbel_sample, axis=2)

                        # [batch_size, max_time, 1]
                        train_output_gumbel_sample_tree2nl_ids = tf.batch_gather(params=tile_tree2nl_ids, indices=expand_train_output_gumbel_sample)

                else:
                    train_output_gumbel_sample_tree2nl_ids = tf.no_op()

            # print_op = tf.print("decoder_logits_train-shape:", tf.shape(decoder_logits_train), "\n")
            # print_op2 = tf.print("max decoder train length:", tf.reduce_max(decoder_train_lengths), "\n")
            # print_op3 = tf.print("decoder_train_inputs_embedded-shape:", tf.shape(decoder_train_inputs_embedded), "\n")

            # with tf.control_dependencies([print_op, print_op2, print_op3]):
            else:
                train_copy_vocab_probs = tf.no_op()

                train_sum_node_probs = tf.no_op()

                train_classifier_gumbel_sample = tf.no_op()

                train_output_gumbel_sample = tf.no_op()

                train_copy_output_sample_probs = tf.no_op()

                train_output_gumbel_sample_tree2nl_ids = tf.no_op()

            train_sample_id = tf.argmax(decoder_logits_train, axis=-1, output_type=tf.int32)

            return sample_id, \
                   decoder_logits_infer, \
                   decoder_logits_train, \
                   train_sample_id, \
                   train_copy_vocab_probs, \
                   train_sum_node_probs, \
                   train_classifier_gumbel_sample, \
                   train_output_gumbel_sample, \
                   train_copy_output_sample_probs, \
                   train_output_gumbel_sample_tree2nl_ids

        return sample_id, \
               decoder_logits_infer, \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op(), \
               tf.no_op()


class Tree2SeqModel:

    def __init__(self, params):

        with tf.variable_scope('embedding'):

            pad_embedding = tf.zeros(shape=[1, params.embed_size], dtype=tf.float32)

            self.src_embeddings = tf.get_variable(name="src_embeddings", shape=[params.src_vocab_size-1, params.embed_size], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            self.tgt_embeddings = tf.get_variable(name="tgt_embeddings", shape=[params.tgt_vocab_size-1, params.embed_size], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            self.src_embeddings = tf.concat([pad_embedding, self.src_embeddings], axis=0)
            self.tgt_embeddings = tf.concat([pad_embedding, self.tgt_embeddings], axis=0)

            self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=params.embed_size, name="encode_lstm_cell")

        with tf.variable_scope('encoder'):

            if params.use_type:
                self.encoder = TreeLSTMEncoder(params.init_weight, params.embed_size, params.hidden_size, max_child_num=params.max_child_num, type_num=params.type_vocab_size, use_source_seq=params.use_source_seq, bidirectional=params.bidirectional, use_type=params.use_type, layer_norm=params.layer_norm)
            else:
                self.encoder = TreeLSTMEncoder(params.init_weight, params.embed_size, params.hidden_size, max_child_num=params.max_child_num, use_source_seq=params.use_source_seq, bidirectional=params.bidirectional, use_type=params.use_type, layer_norm=params.layer_norm)

            if params.use_source_seq:
                # self.combine_t_lstm = NAryTreeLSTM(params.init_weight, params.hidden_size, params.hidden_size, layer_norm=params.layer_norm, prefix="combine_tree_lstm")
                self.combine_linear_h = tf.layers.Dense(units=params.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="combine_tree_dense_h")
                self.combine_linear_c = tf.layers.Dense(units=params.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="combine_tree_dense_c")

        with tf.variable_scope('dense_layer'):
            self.projection_layer = tf.layers.Dense(units=params.tgt_vocab_size, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="projection_layer")

        with tf.variable_scope('decoder'):

            tgt_with_src_in_tgt_unk_vocab_size = None

            if params.copy:
                tgt_with_src_in_tgt_unk_vocab_size = params.s_with_tree_in_seq_unk_vocab_size
            # hidden_size, num_layers, tgt_embeddings, tgt_vocab_size, projection_layer, mode
            # self.attention_layer = tf.layers.Dense(units=params.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.tanh, name="attention_layer")
            self.decoder = LSTMDecoder(params.hidden_size, params.decoder_num_layers, self.tgt_embeddings, params.tgt_vocab_size, self.projection_layer, params.mode, classifier=params.classifier, temperature=params.temperature, layer_norm=params.layer_norm, copy=params.copy, tgt_with_src_in_tgt_unk_vocab_size=tgt_with_src_in_tgt_unk_vocab_size)

        if params.copy:
            with tf.variable_scope('copy'):
                # if params.classifier == "gumbel":
                #     act = None
                # else:
                #     # hard & soft classifier
                #     act = tf.nn.softmax

                act = tf.nn.softmax

                self.copy_vocab_probs_layer = tf.layers.Dense(units=2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="copy_vocab_probs_layer", activation=act)

                self.node_pointer_net = PointerNet()
                self.node_value_pointer_net = PointerNet()

        self.params = params

        self.placeholders = self.create_placeholders(params)

        self.sos_id = self.params.seq_word2idx[self.params.start_token]
        self.eos_id = self.params.seq_word2idx[self.params.end_token]
        self.unk_id = self.params.seq_word2idx[self.params.unk_token]

        if params.reward == "bleu":
            self.reward_fn = bleu_reward_shape
        elif params.reward == "rouge":
            self.reward_fn = rouge_reward_shape
        else:
            raise ValueError("the reward type %s does not exist" % (params.reward))

        # with tf.variable_scope('optimizer'):
        #     self.optimizer = tf.train.AdamOptimizer(params.learning_rate)

    def create_placeholders(self, params):
        placeholders = {}

        placeholders['tree_data'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="tree_data") # [batch_size, max_node_count, max_value_len]
        placeholders['node_value_lens'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="node_value_lens") # [batch_size, max_node_count]
        # placeholders['left_child_ids'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="left_childs") # [batch_size, max_node_count]
        # placeholders['right_child_ids'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="right_childs") # [batch_size, max_node_count]
        placeholders['tree_node_counts'] = tf.placeholder(dtype=tf.int32, shape=[None], name="tree_node_counts") # [batch_size]

        placeholders['child_ids'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="child_ids") # [batch_size, max_node_count, max_child_num]

        placeholders['seq_in_data'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="seq_in_data") # [batch_size, max_seq_len]
        placeholders['seq_out_data'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="seq_out_data") # [batch_size, max_seq_len]
        placeholders['seq_lens'] = tf.placeholder(dtype=tf.int32, shape=[None], name="seq_lens") # [batch_size]

        # if params.use_type:
        placeholders['parent_types'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="parent_types") # [batch_size, max_node_count]
        # placeholders['left_child_types'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="left_child_types") # [batch_size, max_node_count]
        # placeholders['right_child_types'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="right_child_types") # [batch_size, max_node_count]

        placeholders['child_types'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="child_types") # [batch_size, max_node_count, max_child_num]

        if params.use_source_seq:
            placeholders['tree_seq_in_data'] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tree_seq_in_data") # [batch_size, max_seq_len]
            placeholders['tree_seq_lens'] = tf.placeholder(dtype=tf.int32, shape=[None], name="tree_seq_lens") # [batch_size]

        if params.copy:

            # [batch_size, max_node_count]
            # to mask padding nodes and the non-primitive (reduntant) nodes, just keep primitive node which we want to copy
            # (training & testing)
            placeholders['encoder_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="encoder_mask")

            # [batch_size, max_node_count, max_node_value_len]
            # (training & testing)
            placeholders['encoder_value_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="encoder_value_mask")

            # [batch_size, max_time, max_node_count]
            # (training & testing)
            placeholders['tgt_encoder_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="tgt_encoder_mask")

            # [batch_size, max_time, max_node_count, max_node_value_len]
            # (training & testing)
            placeholders['tgt_encoder_value_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name="tgt_encoder_value_mask")

            # # [batch_size, max_time, max_node_count, max_node_value_len, max_node_value_len]
            # # (testing)
            # placeholders['infer_tgt_encoder_value_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, None], name="infer_tgt_encoder_value_mask")

            # [batch_size, max_node_count, max_value_len]
            placeholders['tree2nl_ids'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="tree2nl_ids")

            if params.classifier == "rl":
                placeholders['rl_train'] = tf.placeholder(dtype=tf.bool, shape=[], name="rl_train")
                placeholders['reward'] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="reward") # [batch_size, max_time]


        return placeholders

    def built(self):
        built = self.lstm.built and self.projection_layer.built and self.encoder.built() and self.decoder.built()
        return built


    def train(self, sess, batch, reward_sess=None, saver=None, checkpoint_prefix=None):

        # t = np.array(batch['batch_seq_in_data'])
        # t2 = np.array(batch['batch_seq_out_data'])
        # t3 = np.array(batch['batch_tree_data'])
        # t4 = np.array(batch['batch_node_lens'])
        # t5 = np.array(batch['batch_left_child_ids'])
        # t6 = np.array(batch['batch_right_child_ids'])
        # t7 = np.array(batch['batch_tree_node_counts'])
        # t8 = np.array(batch['batch_seq_lens'])
        # t9 = np.array(batch['batch_parent_types'])
        # t10 = np.array(batch['batch_left_child_types'])
        # t11 = np.array(batch['batch_right_child_types'])
        # print(t.shape, t2.shape, t3.shape, t4.shape, t5.shape, t6.shape, t7.shape, t8.shape, t9.shape, t10.shape, t11.shape, end="\n")

        feed_dict={
            self.placeholders['tree_data']:batch['batch_tree_data'],
            self.placeholders['node_value_lens']: batch['batch_node_lens'],
            # self.placeholders['left_child_ids']: batch['batch_left_child_ids'],
            # self.placeholders['right_child_ids']: batch['batch_right_child_ids'],
            self.placeholders['child_ids']: batch['batch_child_ids'],
            self.placeholders['tree_node_counts']: batch['batch_tree_node_counts'],
            self.placeholders['seq_in_data']: batch['batch_seq_in_data'],
            self.placeholders['seq_out_data']: batch['batch_seq_out_data'],
            self.placeholders['seq_lens']: batch['batch_seq_lens'],

            self.placeholders['parent_types']: batch['batch_parent_types'],
            # self.placeholders['left_child_types']: batch['batch_left_child_types'],
            # self.placeholders['right_child_types']: batch['batch_right_child_types'],
            self.placeholders['child_types']: batch['batch_child_types'],
        }

        if self.params.use_source_seq:
            feed_dict[self.placeholders['tree_seq_in_data']] = batch['batch_tree_seq_in_data']
            feed_dict[self.placeholders['tree_seq_lens']] = batch['batch_tree_seq_lens']

        if self.params.copy:
            feed_dict[self.placeholders['encoder_mask']] = batch['batch_encoder_mask']
            feed_dict[self.placeholders['encoder_value_mask']] = batch['batch_encoder_value_mask']
            feed_dict[self.placeholders['tgt_encoder_mask']] = batch['batch_tgt_encoder_mask']
            feed_dict[self.placeholders['tgt_encoder_value_mask']] = batch['batch_tgt_encoder_value_mask']
            feed_dict[self.placeholders['tree2nl_ids']] = batch['batch_tree2nl_ids']

        # if self.params.classifier == "rl":
        #     feed_dict[self.placeholders['rl_train']] = batch['rl_train']

        if self.params.classifier != "rl":
            loss, train_total_loss, train_sample_id, _ = sess.run([self.loss, self.train_total_loss, self.train_sample_id, self.train_op], feed_dict=feed_dict)
        else:

            feed_dict[self.placeholders['rl_train']] = batch['rl_train']

            # when sample ids, this is unused, so use a fake reward
            feed_dict[self.placeholders['reward']] = [[1]]

            # reward session restore weight from checkpoint, to reuse the weights, whatever in pretraining stage or rl stage
            saver.restore(sess=reward_sess, save_path=checkpoint_prefix)

            if batch['rl_train']:

                # use reward session to get sample ids in order to get the sample ids when in main session training
                # numpy array, [batch_size, max_time]
                train_final_sample_ids = reward_sess.run(self.train_final_sample_ids, feed_dict=feed_dict)

                # list
                train_final_sample_ids_list = train_final_sample_ids.tolist()

                batch_size = train_final_sample_ids.shape[0]
                max_len = train_final_sample_ids.shape[1]

                # reward = bleu_reward(batch_sample_id=train_final_sample_ids_list,
                #             batch_target=batch['batch_gold_seqs'],
                #             params=self.params,
                #             eos_id=self.eos_id,
                #             max_len=max_len)

                reward = self.reward_fn(
                    batch_sample_id=train_final_sample_ids_list,
                    batch_target=batch['batch_gold_seqs'],
                    params=self.params,
                    eos_id=self.eos_id,
                    batch_size=batch_size,
                    max_len=max_len,
                )

                feed_dict[self.placeholders['reward']] = reward

                loss, train_total_loss, train_sample_id, _, train_final_sample_ids2 = sess.run([self.loss, self.train_total_loss, self.train_sample_id, self.train_op, self.train_final_sample_ids], feed_dict=feed_dict)
                assert np.all(train_final_sample_ids==train_final_sample_ids2), "reward session and main session sample is not True"

                saver.save(sess, save_path=checkpoint_prefix)
            else:

                # use reward session to get sample ids in order to get the sample ids when in main session training
                # numpy array, [batch_size, max_time]
                train_final_sample_ids = reward_sess.run(self.train_final_sample_ids, feed_dict=feed_dict)

                loss, train_total_loss, train_sample_id, _, train_final_sample_ids2 = sess.run([self.loss, self.train_total_loss, self.train_sample_id, self.train_op, self.train_final_sample_ids], feed_dict=feed_dict)
                saver.save(sess, save_path=checkpoint_prefix)
                assert np.all(train_final_sample_ids == train_final_sample_ids2), "reward session and main session sample is not True"

        return loss, train_total_loss, train_sample_id

    def predict(self, sess, batch):

        feed_dict = {
            self.placeholders['tree_data']: batch['batch_tree_data'],
            self.placeholders['node_value_lens']: batch['batch_node_lens'],
            # self.placeholders['left_child_ids']: batch['batch_left_child_ids'],
            # self.placeholders['right_child_ids']: batch['batch_right_child_ids'],
            self.placeholders['child_ids']: batch['batch_child_ids'],
            self.placeholders['tree_node_counts']: batch['batch_tree_node_counts'],
            self.placeholders['seq_in_data']: batch['batch_seq_in_data'],
            self.placeholders['seq_out_data']: batch['batch_seq_out_data'],
            self.placeholders['seq_lens']: batch['batch_seq_lens'],

            self.placeholders['parent_types']: batch['batch_parent_types'],
            # self.placeholders['left_child_types']: batch['batch_left_child_types'],
            # self.placeholders['right_child_types']: batch['batch_right_child_types'],
            self.placeholders['child_types']: batch['batch_child_types'],
        }

        if self.params.use_source_seq:
            feed_dict[self.placeholders['tree_seq_in_data']] = batch['batch_tree_seq_in_data']
            feed_dict[self.placeholders['tree_seq_lens']] = batch['batch_tree_seq_lens']

        if self.params.copy:
            feed_dict[self.placeholders['encoder_mask']] = batch['batch_encoder_mask']
            feed_dict[self.placeholders['encoder_value_mask']] = batch['batch_encoder_value_mask']
            feed_dict[self.placeholders['tgt_encoder_mask']] = batch['batch_tgt_encoder_mask']
            feed_dict[self.placeholders['tgt_encoder_value_mask']] = batch['batch_tgt_encoder_value_mask']
            feed_dict[self.placeholders['tree2nl_ids']] = batch['batch_tree2nl_ids']

            if self.params.classifier == "rl":
                feed_dict[self.placeholders['rl_train']] = False
                feed_dict[self.placeholders['reward']] = [[1]]

        sample_id, infer_loss, infer_total_loss = sess.run([self.sample_id, self.infer_loss, self.infer_total_loss], feed_dict=feed_dict)

        return sample_id, infer_loss, infer_total_loss

    def with_infer_loss(self, decoder_infer_logits, labels, decoder_infer_target_lengths):
        '''
        :param decoder_infer_logits: already softmax, [batch_size, max_predict_len, tgt_vocab_size]
        :param labels: [batch_size, max_target_len]
        :param decoder_infer_target_lengths: [batch_size]
        :return:
        '''

        batch_size = tf.shape(labels)[0]

        max_predict_len = tf.shape(decoder_infer_logits)[1]
        max_target_len = tf.shape(labels)[1]

        max_infer_len = tf.cond(max_predict_len > max_target_len, true_fn=lambda : max_target_len, false_fn=lambda : max_predict_len)

        max_infer_lens = tf.cast(max_infer_len * tf.ones_like(decoder_infer_target_lengths), tf.int32)

        # [batch_size] values >= max_infer_len
        decoder_mask_lengths = tf.where(decoder_infer_target_lengths>max_infer_lens, x=max_infer_lens, y=decoder_infer_target_lengths)

        # [batch_size, max_infer_len]
        sequence_mask = tf.sequence_mask(lengths=decoder_mask_lengths, maxlen=max_infer_len, dtype=tf.float32)

        # [batch_size, max_infer_len, 1]
        labels = tf.expand_dims(labels[:, :max_infer_len], axis=2)

        # [batch_size, max_infer_len, tgt_vocab_size]
        probs = decoder_infer_logits[:, :max_infer_len, :]

        # [batch_size, max_infer_len, 1]
        label_probs = tf.batch_gather(params=probs, indices=labels)

        # [batch_size, max_infer_len]
        label_probs = tf.squeeze(label_probs, axis=2)

        # [batch_size, max_infer_len]
        infer_losses = -tf.log(tf.clip_by_value(label_probs, 1e-10, 1.0))

        # scalar
        infer_loss = tf.reduce_sum(infer_losses * sequence_mask) / tf.to_float(batch_size)

        # scalar
        infer_total_loss = tf.reduce_sum(infer_losses * sequence_mask)

        return infer_loss, infer_total_loss

    def build_graph(self):

        batch_size = tf.shape(self.placeholders['tree_data'])[0]
        max_node_count = tf.shape(self.placeholders['tree_data'])[1]
        max_value_len = tf.shape(self.placeholders['tree_data'])[-1]

        # [batch_size * max_node_count, max_value_len]
        encoder_input_ids = tf.reshape(self.placeholders['tree_data'], shape=[-1, max_value_len])

        # [batch_size * max_node_count]
        node_lens = tf.reshape(self.placeholders['node_value_lens'], shape=[-1])

        # [batch_size * max_node_count, max_value_len, embed_size]
        encoder_input_embeds = tf.nn.embedding_lookup(self.src_embeddings, encoder_input_ids)

        if self.params.use_source_seq:
            # [batch_size, max_seq_len, embed_size]
            tree_seq_embeds = tf.nn.embedding_lookup(self.src_embeddings, self.placeholders['tree_seq_in_data'])
            tree_seq_lens = self.placeholders['tree_seq_lens']
        else:
            tree_seq_embeds = None
            tree_seq_lens = None

        # outputs=[batch_size*max_node_count, max_value_len, hidden_size]
        # final_state = ([batch_size*max_node_count, hidden_size], [batch_size*max_node_count, hidden_size])
        outputs, final_state = self.encode_text(self.lstm, encoder_input_embeds, node_lens)

        # [batch_size * max_node_count, max_value_len, hidden_size]
        node_value_states = outputs

        # [batch_size, max_node_count, max_value_len, hidden_size]
        node_value_states = tf.reshape(node_value_states, shape=[batch_size, max_node_count, max_value_len, -1])

        x = tf.reshape(final_state.h, shape=[batch_size, max_node_count, -1])

        node_states = tf.zeros(shape=[batch_size, max_node_count, 2, self.params.hidden_size], dtype=tf.float32)

        parent_types = self.placeholders['parent_types']
        # left_child_types = self.placeholders['left_child_types']
        # right_child_types = self.placeholders['right_child_types']

        # [batch_size, max_node_count, max_child_num]
        child_types = self.placeholders['child_types']

        # left_child_ids = self.placeholders['left_child_ids']
        # right_child_ids = self.placeholders['right_child_ids']

        # [batch_size, max_node_count, max_child_num]
        child_ids = self.placeholders['child_ids']

        encoder = self.encoder.encode(x=x,
               node_states=node_states,
               child_ids=child_ids,
               # left_child_ids=left_child_ids,
               # right_child_ids=right_child_ids,
               parent_types=parent_types,
               child_types=child_types,
               # left_child_types=left_child_types,
               # right_child_types=right_child_types,
               tree_seq_in_embeds=tree_seq_embeds,
               tree_seq_lens=tree_seq_lens)

        if self.params.use_source_seq:
            encoder_node_states, encoder_seq_final_state = encoder

            # [batch_size, hidden_size]
            tree_root_node_state_h = encoder_node_states[:, 0, 0, :]
            tree_root_node_state_c = encoder_node_states[:, 0, 1, :]

            # [batch_size, 2*hiddens_size]
            linear_input_h = tf.concat([tree_root_node_state_h, encoder_seq_final_state.h], axis=1)
            linear_input_c = tf.concat([tree_root_node_state_c, encoder_seq_final_state.c], axis=1)

            combine_h = self.combine_linear_h(linear_input_h)

            combine_c = self.combine_linear_c(linear_input_c)

            # zero_input = tf.zeros(shape=[batch_size, self.params.hidden_size], dtype=tf.float32)
            #
            # combine_h, combine_c = self.combine_t_lstm.forward(x=zero_input,
            #                                                    child_h=(tree_root_node_state_h, encoder_seq_final_state.h),
            #                                                    child_c=(tree_root_node_state_c, encoder_seq_final_state.c))

            final_encoder_repre_h, final_encoder_repre_c = combine_h, combine_c
        else:
            encoder_node_states = encoder
            tree_root_node_state_h = encoder_node_states[:, 0, 0, :]
            tree_root_node_state_c = encoder_node_states[:, 0, 1, :]
            final_encoder_repre_h, final_encoder_repre_c = tree_root_node_state_h, tree_root_node_state_c

        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=final_encoder_repre_h, h=final_encoder_repre_c)

        tree_node_counts = self.placeholders['tree_node_counts']

        train_inputs = self.placeholders['seq_in_data']
        decoder_train_lengths = self.placeholders['seq_lens']

        # [batch_size, max_node_count, hidden_size], use the tree node state
        memory = encoder_node_states[:, :, 0, :]

        sos_id = self.params.seq_word2idx[self.params.start_token]
        eos_id = self.params.seq_word2idx[self.params.end_token]
        unk_id = self.params.seq_word2idx[self.params.unk_token]

        # train_inputs, encoder_state, sos_id, eos_id, memory = None, tree_node_counts = None, decoder_train_lengths = None, max_iterations = None, use_attention = True

        pointer_net_data = None

        if self.params.copy:
            pointer_net_data = PointerNetData(encoder_mask=self.placeholders['encoder_mask'],
                     encoder_value_mask=self.placeholders['encoder_value_mask'],
                     tgt_encoder_mask=self.placeholders['tgt_encoder_mask'],
                     tgt_encoder_value_mask=self.placeholders['tgt_encoder_value_mask'],
                     node_states=memory,
                     node_value_states=node_value_states,
                     copy_vocab_probs_layer=self.copy_vocab_probs_layer,
                     node_pointer_net=self.node_pointer_net,
                     node_value_pointer_net=self.node_value_pointer_net,
                     tree2nl_ids=self.placeholders['tree2nl_ids'],
                     copy_decay_keep_prob=self.params.copy_decay_keep_prob)

        # [batch_size, max_seq_len, tgt_vocab_size]
        decoder_res = self.decoder.decode(train_inputs,
                                        encoder_state,
                                        sos_id,
                                        eos_id,
                                        unk_id,
                                        self.placeholders['parent_types'],
                                        memory,
                                        tree_node_counts,
                                        decoder_train_lengths,
                                        max_iterations=self.params.max_iterations,
                                        use_attention=True,
                                        pointer_net_data=pointer_net_data,
                                        copy_once=self.params.copy_once,
                                        predict_copy_once=self.params.predict_copy_once,
                                        copy_decay=self.params.copy_decay)

        self.sample_id = decoder_res[0]
        self.decoder_logits_infer = decoder_res[1]

        self.decoder_logits_train = decoder_res[2]
        self.train_sample_id = decoder_res[3]

        # [batch_size, max_time, 2] or no_op
        self.train_copy_vocab_probs = decoder_res[4]

        # [batch_size, max_time] or no_op
        self.train_sum_node_probs = decoder_res[5]


        # [batch_size, max_time]
        self.train_copy_gumbel_sample = decoder_res[7]

        # [batch_size, max_time, 1]
        self.train_copy_output_sample_probs = decoder_res[8]

        # [batch_size, max_time, 1]
        train_output_gumbel_sample_tree2nl_ids = decoder_res[9]

        if self.params.classifier == "rl" and not isinstance(train_output_gumbel_sample_tree2nl_ids, tf.Operation):
            # [batch_size, max_time]
            self.train_output_gumbel_sample_tree2nl_ids = tf.squeeze(train_output_gumbel_sample_tree2nl_ids, axis=2)

        # # [batch_size, max_time, 2] or no_op
        # self.train_gumbel_sample = decoder_res[6]

        # [batch_size, max_time] or no_op
        train_classifier_gumbel_sample = decoder_res[6]

        if (self.params.classifier == "gumbel" or self.params.classifier == "rl") and not isinstance(train_classifier_gumbel_sample, tf.Operation):
            # [batch_size, max_time, 2]
            self.train_classifier_gumbel_sample = tf.one_hot(indices=train_classifier_gumbel_sample, depth=2, axis=-1, dtype=tf.float32)

        if self.params.mode == "train":
            self.loss, self.train_total_loss = self.calculate_loss(decoder_train_lengths)
            self.train_op = self.optimization(self.loss)

        # when in 'train' mode, the decoder_train_lengths and decoder_infer_target_lengths is the same,
        # corresponding to placeholder['seq_lens']
        decoder_infer_target_lengths = self.placeholders['seq_lens']

        decoder_infer_logits = self.decoder_logits_infer

        # when in 'train' mode, the 'seq_out_data' can represent the training labels and testing labels
        infer_labels = self.placeholders['seq_out_data']

        self.infer_loss, self.infer_total_loss = self.with_infer_loss(decoder_infer_logits, infer_labels, decoder_infer_target_lengths)

    def calculate_loss(self, decoder_train_lengths):
        '''
        :param decoder_train_lengths: [batch_size]
        :return:
        '''

        labels = self.placeholders['seq_out_data']

        batch_size = tf.shape(labels)[0]
        maxlen = tf.shape(labels)[1]

        # [batch_size, max_time]
        sequence_mask = tf.sequence_mask(lengths=decoder_train_lengths, maxlen=maxlen, dtype=tf.float32)

        # [batch_size, max_time, tgt_vocab_size]
        tgt_vocab_probs = tf.nn.softmax(self.decoder_logits_train, axis=-1)

        # [batch_size, max_time]
        train_gen_gumbel_sample = gumbel_trick(p=tgt_vocab_probs, shape=tf.shape(tgt_vocab_probs), temperature=self.params.temperature)

        # [batch_size, max_time, 1]
        labels = tf.expand_dims(labels, axis=2)

        # [batch_size, max_time, 1]
        tgt_vocab_label_probs = tf.batch_gather(params=tgt_vocab_probs, indices=labels)

        # [batch_size, max_time]
        self.tgt_vocab_label_probs = tf.squeeze(tgt_vocab_label_probs, axis=2)

        if self.params.copy:
            # [batch_size, max_time, max_node_count]
            tgt_encoder_mask = self.placeholders['tgt_encoder_mask']

            # [batch_size, max_time]
            copy_mask = tf.cast(tf.reduce_any(tgt_encoder_mask > 0, axis=-1), tf.float32)

            # [batch_size, max_time]
            gen_mask = 1.0 - copy_mask

            # # [batchs_size, max_time]
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
            #                                                         logits=self.decoder_logits_train)

            copy_prob_max_i = tf.argmax(tf.reduce_max(self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask, axis=-1), output_type=tf.int32)

            copy_prob_max_j = tf.argmax((self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask)[copy_prob_max_i, :], output_type=tf.int32)

            correspond_vocab_gen_prob_copy_max = (self.train_copy_vocab_probs[:, :, 0]*self.tgt_vocab_label_probs)[copy_prob_max_i, copy_prob_max_j]

            correspond_copy_vocab_probs = self.train_copy_vocab_probs[copy_prob_max_i, copy_prob_max_j, :]

            copy_prob_min_i = tf.argmin(tf.reduce_min(self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask, axis=-1), output_type=tf.int32)

            copy_prob_min_j = tf.argmin((self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask)[copy_prob_min_i, :], output_type=tf.int32)

            correspond_vocab_gen_prob_copy_min = (self.train_copy_vocab_probs[:, :, 0]*self.tgt_vocab_label_probs)[copy_prob_min_i, copy_prob_min_j]

            print_op = tf.print("train_copy_vocab_probs:", self.train_copy_vocab_probs)
            print_op2 = tf.print("vocab_gen_probs max:", tf.reduce_max(self.train_copy_vocab_probs[:, :, 0]*self.tgt_vocab_label_probs*gen_mask))
            print_op3 = tf.print("copy probs max:", tf.reduce_max(self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask))
            print_op4 = tf.print("tgt_vocab_label_probs:", self.tgt_vocab_label_probs)
            print_op5 = tf.print("train_sum_node_probs:", self.train_sum_node_probs)
            print_op6 = tf.print("any not 0 prob:", tf.reduce_sum(tf.cast(tf.reduce_any(self.train_sum_node_probs>0, axis=-1), tf.float32)))
            print_op7 = tf.print("copy prob > 0.1:", tf.reduce_sum(tf.cast(tf.reduce_any(self.train_copy_vocab_probs[:, :, 1] > 0.1, axis=-1), tf.float32)))
            print_op8 = tf.print("token can copy", tf.reduce_sum(tf.cast(tf.reduce_any(copy_mask > 0, axis=-1), tf.float32)))
            print_op9 = tf.print("vocab gen max probs:", tf.reduce_max(self.tgt_vocab_label_probs, axis=-1))
            print_op10 = tf.print("copy prob max:", tf.reduce_max(self.train_copy_vocab_probs[:, :, 1]))
            print_op11 = tf.print("correspond vocab gen prob (copy max):", correspond_vocab_gen_prob_copy_max)
            print_op12 = tf.print("copy probs min:", tf.reduce_min(self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask))
            print_op13 = tf.print("correspond vocab gen prob (copy min):", correspond_vocab_gen_prob_copy_min)
            print_op14 = tf.print("correspond_copy_vocab_probs:", correspond_copy_vocab_probs)

            with tf.control_dependencies([print_op3, print_op11, print_op14, print_op12, print_op13]):
                # objective_probs = self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs

                # objective_probs = self.tgt_vocab_label_probs*gen_mask + self.train_sum_node_probs*copy_mask

                assert self.params.classifier in ['hard', 'soft', 'gumbel', 'rl'], "the argument classifier is not in ['hard', 'soft', 'gumbel', 'rl']"

                if self.params.classifier == "hard":
                    objective_probs = self.train_copy_vocab_probs[:, :, 0]*self.tgt_vocab_label_probs*gen_mask + self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs*copy_mask

                    losses = -tf.log(tf.clip_by_value(objective_probs, 1e-10, 1.0))
                elif self.params.classifier == "soft":
                    objective_probs = self.train_copy_vocab_probs[:, :, 0]*self.tgt_vocab_label_probs + self.train_copy_vocab_probs[:, :, 1]*self.train_sum_node_probs

                    losses = -tf.log(tf.clip_by_value(objective_probs, 1e-10, 1.0))
                elif self.params.classifier == "gumbel":
                    # use gumbel softmax sampling, use gumbel sample to replace the functionality of mask
                    # objective_probs = self.tgt_vocab_label_probs * self.train_gumbel_sample[:, :, 0] + self.train_sum_node_probs * self.train_gumbel_sample[:, :, 1]

                    # objective_probs = self.tgt_vocab_label_probs * self.train_gumbel_sample[:, :, 0]*gen_mask + self.train_sum_node_probs * self.train_gumbel_sample[:, :, 1]*copy_mask

                    # objective_probs = self.tgt_vocab_label_probs * self.train_gumbel_sample[:, :, 0] * self.train_copy_vocab_probs[:, :, 0] * gen_mask + self.train_sum_node_probs * self.train_gumbel_sample[:, :, 1] * self.train_copy_vocab_probs[:, :, 1] * copy_mask

                    # objective_probs = self.tgt_vocab_label_probs * self.train_gumbel_sample[:, :, 0] * self.train_copy_vocab_probs[:, :, 0] + self.train_sum_node_probs * self.train_gumbel_sample[:, :, 1] * self.train_copy_vocab_probs[:, :, 1]

                    # losses = -tf.log(tf.clip_by_value(objective_probs, 1e-10, 1.0))

                    # use classifier rl objective
                    # [batch_size, max_time, 2]
                    cls_reward, out_reward = classifier_reward(sample=self.train_classifier_gumbel_sample, gen_mask=gen_mask, copy_mask=copy_mask)

                    losses = -out_reward[:, :, 0] * tf.log(tf.clip_by_value(self.tgt_vocab_label_probs, 1e-10, 1.0)) \
                             -out_reward[:, :, 1] * tf.log(tf.clip_by_value(self.train_sum_node_probs, 1e-10, 1.0)) \
                             -cls_reward[:, :, 0] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) \
                             -cls_reward[:, :, 1] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0))

                else:


                    def rl_train_fn():
                        # use rl, e.t. use gumbel trick to sample from classifier distribution and output distribution
                        # use classifier rl objective
                        # [batch_size, max_time, 2]
                        cls_reward, out_reward = total_reward(classifier_sample=self.train_classifier_gumbel_sample,
                                                              copy_output_sample=self.train_copy_gumbel_sample,
                                                              gen_output_sample=train_gen_gumbel_sample,
                                                              copy_mask=copy_mask,
                                                              gen_mask=gen_mask,
                                                              tgt_encoder_mask=tgt_encoder_mask,
                                                              labels=labels)

                        # [batch_size, max_time, 1]
                        expand_train_gen_gumbel_sample = tf.expand_dims(train_gen_gumbel_sample, axis=2)

                        # [batch_size, max_time]
                        gen_output_sample_probs = tf.squeeze(tf.batch_gather(params=tgt_vocab_probs, indices=expand_train_gen_gumbel_sample), axis=2)

                        # [batch_size, max_time]
                        copy_output_sample_probs = tf.squeeze(self.train_copy_output_sample_probs, axis=2)

                        losses = -out_reward[:, :, 0] * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0)) \
                                 - out_reward[:, :, 1] * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0)) \
                                 - cls_reward[:, :, 0] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) \
                                 - cls_reward[:, :, 1] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0))

                        return losses, tf.zeros(shape=[1, 1], dtype=tf.int32)

                    def rl_train_fn2():
                        # use rl, e.t. use gumbel trick to sample from classifier distribution and output distribution
                        # use classifier rl objective
                        # [batch_size, max_time, 2]
                        cls_reward = classifier_reward2(sample=self.train_classifier_gumbel_sample, gen_mask=gen_mask, copy_mask=copy_mask)

                        # [batch_size, max_time, 1]
                        expand_train_gen_gumbel_sample = tf.expand_dims(train_gen_gumbel_sample, axis=2)

                        # [batch_size, max_time]
                        gen_output_sample_probs = tf.squeeze(tf.batch_gather(params=tgt_vocab_probs, indices=expand_train_gen_gumbel_sample), axis=2)

                        # [batch_size, max_time]
                        copy_output_sample_probs = tf.squeeze(self.train_copy_output_sample_probs, axis=2)

                        # [batch_size, max_time]
                        final_sample_ids = tf.where(self.train_classifier_gumbel_sample[:, :, 0] > self.train_classifier_gumbel_sample[:, :, 1], x=train_gen_gumbel_sample, y=self.train_output_gumbel_sample_tree2nl_ids)

                        # [batch_size, max_time]
                        bleu_reward = self.placeholders['reward']

                        # # use different rewards in output and classifier
                        # losses = -bleu_reward * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0))*self.train_classifier_gumbel_sample[:, :, 0] \
                        #          -bleu_reward * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0))*self.train_classifier_gumbel_sample[:, :, 1] \
                        #          - cls_reward[:, :, 0] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) \
                        #          - cls_reward[:, :, 1] * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0))


                        # use same rewards in output and classifier
                        losses = -bleu_reward * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 0] \
                                 - bleu_reward * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 1] \
                                 - bleu_reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 0] \
                                 - bleu_reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 1]

                        # gen_reward = bleu_reward*cls_reward[:, :, 0]
                        # copy_reward = bleu_reward*cls_reward[:, :, 1]

                        # # [batch_size, max_time]
                        # reward = tf.tile(tf.reduce_mean(gen_reward + copy_reward, axis=-1, keepdims=True), multiples=[1, maxlen])
                        #
                        # losses = - reward * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 0] \
                        #          - reward * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 1] \
                        #          - reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 0] \
                        #          - reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0)) * self.train_classifier_gumbel_sample[:, :, 1]

                        # losses = - gen_reward * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0)) \
                        #          - copy_reward * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0)) \
                        #          - gen_reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) \
                        #          - copy_reward * tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0))

                        # losses = -bleu_reward * tf.log(tf.clip_by_value(gen_output_sample_probs, 1e-10, 1.0)) * gen_mask \
                        #          - bleu_reward * tf.log(tf.clip_by_value(copy_output_sample_probs, 1e-10, 1.0)) * copy_mask \
                        #          - tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 0], 1e-10, 1.0)) * gen_mask \
                        #          - tf.log(tf.clip_by_value(self.train_copy_vocab_probs[:, :, 1], 1e-10, 1.0)) * copy_mask

                        return losses, final_sample_ids

                    def cross_entropy_train_fn():
                        objective_probs = self.train_copy_vocab_probs[:, :, 0] * self.tgt_vocab_label_probs * gen_mask + self.train_copy_vocab_probs[:, :, 1] * self.train_sum_node_probs * copy_mask

                        losses = -tf.log(tf.clip_by_value(objective_probs, 1e-10, 1.0))

                        return losses, tf.zeros(shape=[1, 1], dtype=tf.int32)

                    if self.params.mixed_train_weight is None:
                        print("train just using RL loss...")
                        # [batch_size, max_time], [batch_size, max_time]
                        losses, train_final_sample_ids = tf.cond(self.placeholders['rl_train'], true_fn=lambda: rl_train_fn2(), false_fn=lambda: cross_entropy_train_fn())
                    else:
                        print("train maxing MLE loss and RL loss...")
                        mle_losses, _ = cross_entropy_train_fn()
                        rl_losses, train_final_sample_ids = rl_train_fn2()

                        losses = self.params.mixed_train_weight * mle_losses + (1.0 - self.params.mixed_train_weight)*rl_losses

                    self.train_final_sample_ids = train_final_sample_ids
        else:

            # no copy ,just vocab gen
            objective_probs = self.tgt_vocab_label_probs

            losses = -tf.log(tf.clip_by_value(objective_probs, 1e-10, 1.0))

        train_total_loss = tf.reduce_sum(losses * sequence_mask)

        loss = tf.reduce_sum(losses * sequence_mask) / tf.to_float(batch_size)

        if self.params.l2_loss:

            for v in self.projection_layer.variables:

                if 'kernel' in v.name:
                    loss += self.params.l2_weight * tf.nn.l2_loss(v)

        return loss, train_total_loss

    def optimization(self, loss):

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        # for p in params:
        #     print("variable:", p)

        print_ops = []


        # for g, p in zip(gradients, params):
        #     print_ops .append(tf.print("gradients:", p.name, tf.shape(g), "\n"))

        # for g, p in zip(gradients, params):
        #     print("g:", g)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params.clip_grad)

        with tf.control_dependencies(print_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)

            train_op = optimizer.apply_gradients(zip(gradients, params))

        return train_op

    def encode_text(self, cell, encoder_input_embeds, node_lens):

        # [batch_size * max_node_count, max_value_len, hidden_size]
        outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=encoder_input_embeds,
                                           sequence_length=node_lens,
                                           dtype=tf.float32,
                                           time_major=False,
                                           )

        return outputs, state




            




