
class BinaryTree:

    def __init__(self, json_data, primitive_types, use_type=False, single_word=False, separator="[SEP]"):

        self.value = json_data['root']

        self.primitive_types = primitive_types

        self.separator = separator

        if self.value is not None:
            self.value = self.value.lower()
            if single_word:
                self.value = separator.join(self.value.strip().split())

        if len(json_data['children']) > 0 and json_data['children'][0]['root'] is not None:
            self.left_child = BinaryTree(json_data['children'][0], use_type=use_type, primitive_types=primitive_types, single_word=single_word, separator=separator)
        else:
            self.left_child = None

        if len(json_data['children']) > 1:
            self.right_child = BinaryTree(json_data['children'][1], use_type=use_type, primitive_types=primitive_types, single_word=single_word, separator=separator)
        else:
            self.right_child = None

        self.type = None

        if use_type:
            self.type = json_data['type']

    def is_leaf(self):

        return self.left_child is None and self.right_child is None

    def get_vocab(self, vocab):

        value = self.value

        words = value.strip().split()

        for w in words:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

        if self.left_child is not None:
            self.left_child.get_vocab(vocab)

        if self.right_child is not None:
            self.right_child.get_vocab(vocab)

    def get_type_vocab(self, vocab):

        if self.type not in vocab:
            vocab[self.type] = 1
        else:
            vocab[self.type] += 1

        if self.left_child is not None:
            self.left_child.get_type_vocab(vocab)

        if self.right_child is not None:
            self.right_child.get_type_vocab(vocab)

    def process_sub_word_in_seq(self, word_list):

        replace_count = 0

        # if self.type is None:
        #     return word_list, replace_count

        if (self.type is not None and self.type in self.primitive_types) or len(self.primitive_types) <= 0 or self.type is None:

            processed_list =[]

            before_idx = -1

            for i in range(len(word_list)):

                if word_list[i] not in self.value:
                    if before_idx >= 0:
                        processed_list.append(self.value)
                        replace_count += 1
                        before_idx = -1

                    processed_list.append(word_list[i])
                else:
                    if i == len(word_list)-1:
                        processed_list.append(self.value)
                        replace_count += 1
                    else:
                        before_idx = i

            result_list = processed_list

        else:
            result_list = word_list

        if self.left_child is not None:
            result_list, child_replace_count = self.left_child.process_sub_word_in_seq(result_list)
            replace_count += child_replace_count

        if self.right_child is not None:
            result_list, child_replace_count = self.right_child.process_sub_word_in_seq(result_list)
            replace_count += child_replace_count

        return result_list, replace_count


class NaryTree:

    def __init__(self, json_data, primitive_types, use_type=False, single_word=False, separator="[SEP]"):

        self.value = json_data['root']

        self.primitive_types = primitive_types

        self.separator = separator

        if self.value is not None:
            self.value = self.value
            if single_word:
                self.value = separator.join(self.value.strip().split())
                # pass

        self.children = []

        for child in json_data['children']:
            if child['root'] is not None:
                child_tree = NaryTree(child, use_type=use_type, primitive_types=primitive_types, single_word=single_word, separator=separator)
                self.children.append(child_tree)

        self.type = None

        if use_type:
            self.type = json_data['type']

    def is_leaf(self):

        return len(self.children) <= 0

    def get_vocab(self, vocab):

        value = self.value

        words = value.strip().split()

        for w in words:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

        for child_tree in self.children:
            child_tree.get_vocab(vocab)

    def to_serializable(self):
        d = {}
        d['value'] = self.value
        d['type'] = self.type
        d['children'] = []

        for child in self.children:
            d['children'].append(child.to_serializable())

        return d

    def get_type_vocab(self, vocab):

        if self.type not in vocab:
            vocab[self.type] = 1
        else:
            vocab[self.type] += 1

        for child_tree in self.children:
            child_tree.get_type_vocab(vocab)

    def process_sub_word_in_seq(self, word_list, replaced_flag_list):

        assert len(word_list) == len(replaced_flag_list), "word_list length must equal replaced_flag_list length"

        value = self.separator.join(self.value.strip().split())

        replace_count = 0

        # if self.type is None:
        #     return word_list, replace_count, replaced_flag_list

        if (self.type is not None and self.type in self.primitive_types) or len(self.primitive_types) <= 0 or self.type is None:

            processed_list =[]

            new_replaced_flag_list = []

            before_idx = -1

            for i in range(len(word_list)):

                if word_list[i] not in value.split(self.separator):

                    if before_idx >= 0:
                        processed_list.append(value)
                        new_replaced_flag_list.append(True)
                        replace_count += 1
                        before_idx = -1

                    processed_list.append(word_list[i])
                    new_replaced_flag_list.append(replaced_flag_list[i])
                else:
                    if not replaced_flag_list[i]:
                        if i >= len(word_list)-1:
                            processed_list.append(value)
                            new_replaced_flag_list.append(True)
                            replace_count += 1
                            before_idx = -1
                        else:
                            before_idx = i
                    else:
                        # in value, while is replaced before
                        processed_list.append(word_list[i])
                        new_replaced_flag_list.append(replaced_flag_list[i])
                        before_idx = -1

            result_list = processed_list
            result_replaced_flag_list = new_replaced_flag_list

        else:
            result_list = word_list
            result_replaced_flag_list = replaced_flag_list

        for child_tree in self.children:
            result_list, child_replace_count, result_replaced_flag_list = child_tree.process_sub_word_in_seq(result_list, result_replaced_flag_list)
            replace_count += child_replace_count

        return result_list, replace_count, result_replaced_flag_list





