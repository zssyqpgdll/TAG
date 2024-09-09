from bleu import compute_bleu

from nmt_rouge import rouge, rouge_n

def bleu_fn(predicts, targets):

    '''
    :param predicts: list of predictive sequence
    :param targets: list of  target sequence
    :return:
    '''

    targets = [[tgt] for tgt in targets]

    return compute_bleu(reference_corpus=targets, translation_corpus=predicts)[0]

def rouge_fn(predicts, targets):
    '''
    :param predicts:
    :param targets:
    :return:
    '''

    predicts = [' '.join(p) for p in predicts]

    targets = [' '.join(tgt) for tgt in targets]

    return rouge(predicts, targets)['rouge_l/f_score']

def bleu_rouge_fn(predicts, targets):

    rouge_predicts = [' '.join(p) for p in predicts]

    rouge_targets = [' '.join(tgt) for tgt in targets]

    rouge_results = rouge(rouge_predicts, rouge_targets)

    bleu_score = bleu_fn(predicts, targets)

    rouge_results['bleu'] = bleu_score

    return rouge_results

def compare_with_best_results(best_result_dict, cur_result_dict, metric='bleu'):

    if metric == 'bleu' or metric == 'both':
        assert 'bleu' in best_result_dict and 'bleu' in cur_result_dict, \
            "bleu does not exist in the best_result_dict:%s or cur_result_dict:%s as a key !" % (best_result_dict.__repr__(), cur_result_dict.__repr__())

        if best_result_dict['bleu'] < cur_result_dict['bleu']:
            best_result_dict['bleu'] = cur_result_dict['bleu']

    if metric == 'rouge' or metric == 'both':
        assert 'rouge_1/f_score' in best_result_dict and 'rouge_1/f_score' in cur_result_dict, \
            "rouge_1/f_score does not exist in the best_result_dict:%s or cur_result_dict:%s as a key !" % (
            best_result_dict.__repr__(), cur_result_dict.__repr__())

        if best_result_dict['rouge_1/f_score'] < cur_result_dict['rouge_1/f_score']:
            best_result_dict['rouge_1/f_score'] = cur_result_dict['rouge_1/f_score']
            best_result_dict['rouge_1/r_score'] = cur_result_dict['rouge_1/r_score']
            best_result_dict['rouge_1/p_score'] = cur_result_dict['rouge_1/p_score']

        assert 'rouge_2/f_score' in best_result_dict and 'rouge_2/f_score' in cur_result_dict, \
            "rouge_2/f_score does not exist in the best_result_dict:%s or cur_result_dict:%s as a key !" % (
                best_result_dict.__repr__(), cur_result_dict.__repr__())

        if best_result_dict['rouge_2/f_score'] < cur_result_dict['rouge_2/f_score']:
            best_result_dict['rouge_2/f_score'] = cur_result_dict['rouge_2/f_score']
            best_result_dict['rouge_2/r_score'] = cur_result_dict['rouge_2/r_score']
            best_result_dict['rouge_2/p_score'] = cur_result_dict['rouge_2/p_score']

        assert 'rouge_l/f_score' in best_result_dict and 'rouge_l/f_score' in cur_result_dict, \
            "rouge_l/f_score does not exist in the best_result_dict:%s or cur_result_dict:%s as a key !" % (
                best_result_dict.__repr__(), cur_result_dict.__repr__())

        if best_result_dict['rouge_l/f_score'] < cur_result_dict['rouge_l/f_score']:
            best_result_dict['rouge_l/f_score'] = cur_result_dict['rouge_l/f_score']
            best_result_dict['rouge_l/r_score'] = cur_result_dict['rouge_l/r_score']
            best_result_dict['rouge_l/p_score'] = cur_result_dict['rouge_l/p_score']