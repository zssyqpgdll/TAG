class DataConfig:

    def __init__(self,
                 embed_size,
                 hidden_size,
                 num_epoch,
                 max_iterations,
                 eval_begin_step,
                 eval_every_steps,
                 max_child_num,
                 copy_decay_keep_prob,
                 pretrain_steps,
                 min_vocab_count,
                 train_file,
                 dev_file,
                 test_file,
                 tree_vocab_file,
                 seq_vocab_file,
                 type_vocab_file,
                 primitive_type_file,
                 checkpoint_dir,
                 rl_restore_checkpoint_dir,
                 eval_result_log,
                 predict_file,
                 save_input_to_file,
                 save_output_to_file,
                 # iter_loss_log,
                 iter_eval_log):

        self.embed_size = embed_size

        self.hidden_size = hidden_size

        self.num_epoch = num_epoch

        self.max_iterations = max_iterations

        self.eval_begin_step = eval_begin_step

        self.eval_every_steps = eval_every_steps

        self.max_child_num = max_child_num

        self.copy_decay_keep_prob = copy_decay_keep_prob

        self.pretrain_steps = pretrain_steps

        self.min_vocab_count = min_vocab_count

        self.train_file = train_file

        self.dev_file = dev_file

        self.test_file = test_file

        self.tree_vocab_file = tree_vocab_file

        self.seq_vocab_file = seq_vocab_file

        self.type_vocab_file = type_vocab_file

        self.primitive_type_file = primitive_type_file

        self.checkpoint_dir = checkpoint_dir

        self.rl_restore_checkpoint_dir = rl_restore_checkpoint_dir

        self.eval_result_log = eval_result_log

        self.predict_file = predict_file

        self.save_input_to_file = save_input_to_file

        self.save_output_to_file = save_output_to_file

        # self.iter_loss_log = iter_loss_log

        self.iter_eval_log = iter_eval_log

def init_data_config(trainset, devset, testset):

    # output dir choice: tree2seq_output  tree2seq_copy_output
    tmp_output_dirs = ["tree2seq_output", "tree2seq_copy_output", "ours", "test_tp_enc_output", "test_tp_rtc_output",
                       "test_cd_output", "test_rl_output", "test_marginal_output"]

    tmp_output_dir = tmp_output_dirs[1]

    wikisql_config = DataConfig(embed_size=128,
                 hidden_size=128,
                 num_epoch=10, #30
                 max_iterations=50,
                 eval_begin_step=1, # 3500
                 eval_every_steps=500, # 500
                 max_child_num=4,
                 copy_decay_keep_prob=0.8,
                 pretrain_steps=-1,
                 min_vocab_count=4,
                 train_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/wikisql_' + trainset + '.json',
                 dev_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/wikisql_' + devset + '.json',
                 test_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/wikisql_' + testset + '.json',
                 tree_vocab_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/tree.vocab',
                 seq_vocab_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/seq.vocab',
                 type_vocab_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/type.vocab',
                 primitive_type_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/primitive_type.txt',
                 checkpoint_dir='../data/wikisql_asdl_original_json_tree_with_type&seq3/ckpt_with_source_seq4',
                 rl_restore_checkpoint_dir='../data/wikisql_asdl_original_json_tree_with_type&seq3/rl_restore_ckpt_with_source_seq4',
                 eval_result_log='../data/wikisql_asdl_original_json_tree_with_type&seq3/eval_result_with_source_seq4.log',
                 predict_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/predict_with_source_seq4.txt',
                 save_input_to_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + "input4",
                 save_output_to_file='../data/wikisql_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + "output4",
                 # iter_loss_log='../data/wikisql_asdl_original_json_tree_with_type&seq3/iter_loss.log',
                 iter_eval_log='../data/wikisql_asdl_original_json_tree_with_type&seq3/' + tmp_output_dir + "/" + 'iter_eval4.log')

    geo_config = DataConfig(embed_size=128,   #gpu: 96
                            hidden_size=128,  #gpu: 96
                            num_epoch=1000,
                            max_iterations=30,
                            eval_begin_step=500,
                            eval_every_steps=100,
                            max_child_num=5,
                            copy_decay_keep_prob=0.5,
                            pretrain_steps=-1,
                            min_vocab_count=4,
                            train_file='../data/geo_asdl_original_json_tree_with_type&seq/geo_' + trainset + '.json',
                            dev_file='../data/geo_asdl_original_json_tree_with_type&seq/geo_' + devset + '.json',
                            test_file='../data/geo_asdl_original_json_tree_with_type&seq/geo_' + testset + '.json',
                            tree_vocab_file='../data/geo_asdl_original_json_tree_with_type&seq/tree.vocab',
                            seq_vocab_file='../data/geo_asdl_original_json_tree_with_type&seq/seq.vocab',
                            type_vocab_file='../data/geo_asdl_original_json_tree_with_type&seq/type.vocab',
                            primitive_type_file='../data/geo_asdl_original_json_tree_with_type&seq/primitive_type.txt',
                            checkpoint_dir='../data/geo_asdl_original_json_tree_with_type&seq/ckpt_with_source_seq',
                            rl_restore_checkpoint_dir='../data/geo_asdl_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq',
                            eval_result_log='../data/geo_asdl_original_json_tree_with_type&seq/eval_result_with_source_seq.log',
                            predict_file='../data/geo_asdl_original_json_tree_with_type&seq/predict_with_source_seq.txt',
                            save_input_to_file='../data/geo_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input",
                            save_output_to_file='../data/geo_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output",
                            # iter_loss_log='../data/geo_asdl_original_json_tree_with_type&seq/iter_loss.log',
                            iter_eval_log='../data/geo_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval.log')

    geo_lamb_config = DataConfig(embed_size=128,
                            hidden_size=128,
                            num_epoch=1000,
                            max_iterations=30,
                            eval_begin_step=500,
                            eval_every_steps=100,
                            max_child_num=6,
                            copy_decay_keep_prob=0.5,
                            pretrain_steps=-1,
                            min_vocab_count=4,
                            train_file='../data/geo_lamb_original_json_tree_with_type&seq/' + trainset + '.json',
                            dev_file='../data/geo_lamb_original_json_tree_with_type&seq/' + devset + '.json',
                            test_file='../data/geo_lamb_original_json_tree_with_type&seq/' + testset + '.json',
                            tree_vocab_file='../data/geo_lamb_original_json_tree_with_type&seq/tree.vocab',
                            seq_vocab_file='../data/geo_lamb_original_json_tree_with_type&seq/seq.vocab',
                            type_vocab_file='../data/geo_lamb_original_json_tree_with_type&seq/type.vocab',
                            primitive_type_file='../data/geo_lamb_original_json_tree_with_type&seq/primitive_type.txt',
                            checkpoint_dir='../data/geo_lamb_original_json_tree_with_type&seq/ckpt_with_source_seq',
                            rl_restore_checkpoint_dir='../data/geo_lamb_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq',
                            eval_result_log='../data/geo_lamb_original_json_tree_with_type&seq/eval_result_with_source_seq.log',
                            predict_file='../data/geo_lamb_original_json_tree_with_type&seq/predict_with_source_seq.txt',
                            save_input_to_file='../data/geo_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input",
                            save_output_to_file='../data/geo_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output",
                            # iter_loss_log='../data/geo_lamb_original_json_tree_with_type&seq/iter_loss.log',
                            iter_eval_log='../data/geo_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval.log')

    atis_config = DataConfig(embed_size=64,   #gpu: 32 cpu: 64
                             hidden_size=64,  #gpu: 32 cpu: 64
                             num_epoch=100,
                             max_iterations=50,
                             eval_begin_step=800,
                             eval_every_steps=100,
                             max_child_num=15,
                             copy_decay_keep_prob=0.4,
                             pretrain_steps=-1,
                             min_vocab_count=4,
                             train_file='../data/atis_asdl_original_json_tree_with_type&seq/atis_' + trainset + '.json',
                             dev_file='../data/atis_asdl_original_json_tree_with_type&seq/atis_' + devset + '.json',
                             test_file='../data/atis_asdl_original_json_tree_with_type&seq/atis_' + testset + '.json',
                             tree_vocab_file='../data/atis_asdl_original_json_tree_with_type&seq/tree.vocab',
                             seq_vocab_file='../data/atis_asdl_original_json_tree_with_type&seq/seq.vocab',
                             type_vocab_file='../data/atis_asdl_original_json_tree_with_type&seq/type.vocab',
                             primitive_type_file='../data/atis_asdl_original_json_tree_with_type&seq/primitive_type.txt',
                             checkpoint_dir='../data/atis_asdl_original_json_tree_with_type&seq/ckpt_with_source_seq4',
                             rl_restore_checkpoint_dir='../data/atis_asdl_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq4',
                             eval_result_log='../data/atis_asdl_original_json_tree_with_type&seq/eval_result_with_source_seq4.log',
                             predict_file='../data/atis_asdl_original_json_tree_with_type&seq/predict_with_source_seq4.txt',
                             save_input_to_file='../data/atis_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input4",
                             save_output_to_file='../data/atis_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output4",
                             # iter_loss_log='../data/atis_asdl_original_json_tree_with_type&seq/iter_loss.log',
                             iter_eval_log='../data/atis_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval4.log')

    atis_lamb_config = DataConfig(embed_size=64,  #cpu: 64
                             hidden_size=64,      #cpu: 64
                             num_epoch=200,
                             max_iterations=50,
                             eval_begin_step=800,
                             eval_every_steps=100,
                             max_child_num=15,
                             copy_decay_keep_prob=0.8,
                             pretrain_steps=-1,
                             min_vocab_count=4,
                             train_file='../data/atis_lamb_original_json_tree_with_type&seq/' + trainset + '.json',
                             dev_file='../data/atis_lamb_original_json_tree_with_type&seq/' + devset + '.json',
                             test_file='../data/atis_lamb_original_json_tree_with_type&seq/' + testset + '.json',
                             tree_vocab_file='../data/atis_lamb_original_json_tree_with_type&seq/tree.vocab',
                             seq_vocab_file='../data/atis_lamb_original_json_tree_with_type&seq/seq.vocab',
                             type_vocab_file='../data/atis_lamb_original_json_tree_with_type&seq/type.vocab',
                             primitive_type_file='../data/atis_lamb_original_json_tree_with_type&seq/primitive_type.txt',
                             checkpoint_dir='../data/atis_lamb_original_json_tree_with_type&seq/ckpt_with_source_seq',
                             rl_restore_checkpoint_dir='../data/atis_lamb_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq',
                             eval_result_log='../data/atis_lamb_original_json_tree_with_type&seq/eval_result_with_source_seq.log',
                             predict_file='../data/atis_lamb_original_json_tree_with_type&seq/predict_with_source_seq.txt',
                             save_input_to_file='../data/atis_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input",
                             save_output_to_file='../data/atis_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output",
                             # iter_loss_log='../data/atis_lamb_original_json_tree_with_type&seq/iter_loss.log',
                             iter_eval_log='../data/atis_lamb_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval.log')

    conala_config = DataConfig(embed_size=64, # use type, cpu: 64
                             hidden_size=64,  # use type, cpu: 64
                             num_epoch=200,
                             max_iterations=15,
                             eval_begin_step=100,
                             eval_every_steps=50,
                             max_child_num=10,
                             copy_decay_keep_prob=0.7,
                             pretrain_steps=-1,
                             min_vocab_count=2,
                             train_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/conala_' + trainset + '.json',
                             dev_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/conala_' + devset + '.json',
                             test_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/conala_' + testset + '.json',
                             tree_vocab_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/tree.vocab',
                             seq_vocab_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/seq.vocab',
                             type_vocab_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/type.vocab',
                             primitive_type_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/primitive_type.txt',
                             checkpoint_dir='../data/conala_py3_asdl_original_json_tree_with_type&seq/ckpt_with_source_seq',
                             rl_restore_checkpoint_dir='../data/conala_py3_asdl_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq',
                             eval_result_log='../data/conala_py3_asdl_original_json_tree_with_type&seq/eval_result_with_source_seq.log',
                             predict_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/predict_with_source_seq.txt',
                             save_input_to_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input",
                             save_output_to_file='../data/conala_py3_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output",
                             # iter_loss_log='../data/conala_py3_asdl_original_json_tree_with_type&seq/iter_loss.log',
                             iter_eval_log='../data/conala_py3_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval.log')

    django_config = DataConfig(embed_size=64,  #use type, cpu: 4
                               hidden_size=64, # use type, cpu: 4
                               num_epoch=200,
                               max_iterations=20,
                               eval_begin_step=500,
                               eval_every_steps=100,
                               max_child_num=59,
                               copy_decay_keep_prob=0.9,
                               pretrain_steps=-1,
                               min_vocab_count=4,
                               train_file='../data/django_py2_asdl_original_json_tree_with_type&seq/django_' + trainset + '.json',
                               dev_file='../data/django_py2_asdl_original_json_tree_with_type&seq/django_' + devset + '.json',
                               test_file='../data/django_py2_asdl_original_json_tree_with_type&seq/django_' + testset + '.json',
                               tree_vocab_file='../data/django_py2_asdl_original_json_tree_with_type&seq/tree.vocab',
                               seq_vocab_file='../data/django_py2_asdl_original_json_tree_with_type&seq/seq.vocab',
                               type_vocab_file='../data/django_py2_asdl_original_json_tree_with_type&seq/type.vocab',
                               primitive_type_file='../data/django_py2_asdl_original_json_tree_with_type&seq/primitive_type.txt',
                               checkpoint_dir='../data/django_py2_asdl_original_json_tree_with_type&seq/ckpt_with_source_seq',
                               rl_restore_checkpoint_dir='../data/django_py2_asdl_original_json_tree_with_type&seq/rl_restore_ckpt_with_source_seq',
                               eval_result_log='../data/django_py2_asdl_original_json_tree_with_type&seq/eval_result_with_source_seq.log',
                               predict_file='../data/django_py2_asdl_original_json_tree_with_type&seq/predict_with_source_seq.txt',
                               save_input_to_file='../data/django_py2_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "input",
                               save_output_to_file='../data/django_py2_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + "output",
                               # iter_loss_log='../data/django_py2_asdl_original_json_tree_with_type&seq/iter_loss.log',
                               iter_eval_log='../data/django_py2_asdl_original_json_tree_with_type&seq/' + tmp_output_dir + "/" + 'iter_eval.log')

    return {'wikisql': wikisql_config, 'geo': geo_config, 'atis': atis_config, 'geo_lamb': geo_lamb_config, 'atis_lamb': atis_lamb_config, "conala": conala_config, "django": django_config}

def get_data_config(dataset_name, trainset='train', devset='dev', testset='test'):

    data_configs = init_data_config(trainset=trainset, devset=devset, testset=testset)

    return data_configs[dataset_name]