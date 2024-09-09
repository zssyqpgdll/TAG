## TAG : Type Auxiliary Guiding for Code Comment Generation

Here we provide the official implementation described in the  ACL 2020 paper [TAG : Type Auxiliary Guiding for Code Comment Generation](https://arxiv.org/abs/2005.02835).

### Dependencies

The code is tested under **Python 3.6**. All dependencies are listed in [requirements.txt](./requirements.txt):
```
numpy
tensorflow-gpu==1.12
```

Run the following commands to clone the repository and install packages:
```bash
git clone https://github.com/DMIRLAB-Group/TAG.git
cd TAG
pip install -r requirements.txt
```

### Dataset

The dataset used contains different programming languages and can be found [here](./data).

The dataset is divided into train, dev and test sets, and each division set is a json file with the format:

```json
{
    "source_ast": {"childern": [xxx, xxx, xxx, ...], "type": "stmt", "root"："Select"},
    "source_prog": ["src_word1", "src_word2", "src_word3", ...], 
    "target_prog": ["tgt_word1", "tgt_word2", "tgt_word3", ...]
}
```

#### Creating Custom Dataset

Instantiate the `DataConfig` class in `config.py` to create a custom dataset:

```python
custom_config = DataConfig(embed_size=128,
                 hidden_size=128,
                 num_epoch=10,
                 max_iterations=50,
                 eval_begin_step=1,
                 eval_every_steps=500,
                 max_child_num=4,
                 copy_decay_keep_prob=0.8,
                 pretrain_steps=-1,
                 min_vocab_count=4,
                 train_file='../data/custom_asdl_original_json_tree_with_type&seq3/wikisql_' + trainset + '.json',
                 dev_file='../data/custom_asdl_original_json_tree_with_type&seq3/wikisql_' + devset + '.json',
                 test_file='../data/custom_asdl_original_json_tree_with_type&seq3/wikisql_' + testset + '.json',
                 tree_vocab_file='../data/custom_asdl_original_json_tree_with_type&seq3/tree.vocab',
                 seq_vocab_file='../data/custom_asdl_original_json_tree_with_type&seq3/seq.vocab',
                 type_vocab_file='../data/custom_asdl_original_json_tree_with_type&seq3/type.vocab',
                 primitive_type_file='../data/custom_asdl_original_json_tree_with_type&seq3/primitive_type.txt',
                 checkpoint_dir='../data/custom_asdl_original_json_tree_with_type&seq3/ckpt_with_source_seq4',
                 rl_restore_checkpoint_dir='../data/custom_asdl_original_json_tree_with_type&seq3/rl_restore_ckpt_with_source_seq4',
                 eval_result_log='../data/custom_asdl_original_json_tree_with_type&seq3/eval_result_with_source_seq4.log',
                 predict_file='../data/custom_asdl_original_json_tree_with_type&seq3/predict_with_source_seq4.txt',
                 save_input_to_file='../data/custom_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + "input4",
                 save_output_to_file='../data/custom_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + "output4",
                 # iter_loss_log='../data/custom_asdl_original_json_tree_with_type&seq3/iter_loss.log',
                 iter_eval_log='../data/custom_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + 'iter_eval4.log')
```

### Quick Start

Modify the content in line `921` of `main.py` to run different dataset:

```python
data_config = get_data_config('atis', trainset='train', devset='dev', testset='test')
```

Choices for dataset name are ['wikisql', 'atis', 'atis_lamb', 'conala', 'geo', 'geo_lamb', 'django'].

#### Training

```bash
cd model
python main.py --mode train
```

The model checkpoints are saved in the directory `ckpt_with_source_seq4/` and `rl_restore_ckpt_with_source_seq4/`.

<!-- See the log file through:
```bash
vim ../data/atis_asdl_original_json_tree_with_type&seq3/tree2seq_copy_output/iter_eval4.log
``` -->

#### Testing

```bash
cd model
pythhon main.py --mode test
```

See the predicted file through:

```bash
vim ../data/atis_asdl_original_json_tree_with_type&seq/tree2seq_copy_output/output4.test
```

An example is provided below:

```bash
Predict: what is the flight from ci0 to ci1 on mn0 dn0
Gold: i'd like also to book a one way flight from ci0 to ci1 the cheapest one on mn0 dn0
Tree: [["Argmin"], ["$0"], ["domain"], ["body"], ["And"], ["Apply"], ["arguments"], ["fare"], ["arguments"], ["Apply"], ["Apply"], ["Apply"], ["Apply"], ["Apply"], ["Apply"], ["Variable"], ["flight"], ["arguments"], ["oneway"], ["arguments"], ["from"], ["arguments"], ["to"], ["arguments"], ["day_number"], ["arguments"], ["month"], ["arguments"], ["$0"], ["Variable"], ["Variable"], ["Variable"], ["Entity"], ["Variable"], ["Entity"], ["Variable"], ["Entity"], ["Variable"], ["Entity"], ["$0"], ["$0"], ["$0"], ["ci0"], ["$0"], ["ci1"], ["$0"], ["dn0"], ["$0"], ["mn0"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"], ["<pad>"]]
Seq In: <sos> i'd like also to book a one way flight from ci0 to ci1 the cheapest one on mn0 dn0 <pad> <pad>
Seq Out: i'd like also to book a one way flight from ci0 to ci1 the cheapest one on mn0 dn0 <eos> <pad> <pad>
```

### Acknowledgement

We modified code from [treelstm](https://github.com/stanfordnlp/treelstm) and [NMT](https://github.com/tensorflow/nmt). We would like to thank the authors of these repositeries.

### Citation

```cite
@inproceedings{cai-etal-2020-tag,
    author = {Cai, Ruichu and Liang, Zhihao and Xu, Boyan and Li, Zijian and Hao, Yuexing and Chen, Yao},
    title = {TAG : Type Auxiliary Guiding for Code Comment Generation},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year = {2020}
}
```


### License

[MIT License](https://opensource.org/licenses/MIT)