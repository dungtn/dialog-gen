"""
This module loads and pre-processes a bAbI dataset into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import json
import collections
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('source_dir', 'datasets/', 'Directory containing  sources.')
tf.app.flags.DEFINE_string('dest_dir', 'datasets/processed/', 'Where to write datasets.')

SPLIT_RE = re.compile('(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0


def int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]


def parse_dialogs(lines, max_length=200):
    """
    Parse the Ubuntu dialog corpus format described here: https://github.com/brmson/dataset-sts/tree/master/data/anssel/ubuntu
    """
    dialogs = []
    for line in lines:
        line, _ = line.decode('utf-8').strip().rsplit(',', 1)

        turns = [ turn for turn in line.split('__eot__') ]

        # take each pair of utterances as a query/answer pair
        for i, (query, answer) in enumerate(zip(turns[1:], turns[2:])):
            query  = filter(lambda e: e != '__eou__', tokenize(query))
            answer = filter(lambda e: e != '__eou__', tokenize(answer))
            if len(query) <= max_length and len(answer) <= max_length:
                dialog = [ tokenize(utterance) for turn in turns[:i+1] for utterance in turn.split('__eou__') ]
                dialogs.append((dialog, query, answer))

    return dialogs


def save_dataset(dialogs, path):
    """
    Save the dialogs into TFRecords.

    NOTE: Since each sentence is a consistent length from padding, we use
    `tf.train.Example`, rather than a `tf.train.SequenceExample`, which is
    _slightly_ faster.
    """
    writer = tf.python_io.TFRecordWriter(path)
    for history, query, answer in dialogs:
        history_flat = [token_id for sentence in history for token_id in sentence]

        features = tf.train.Features(feature={
            'history': int64_features(history_flat),
            'query': int64_features(query),
            'answer': int64_features(answer),
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()


def tokenize_dialogs(dialogs, token_to_id):
    """
    Convert all tokens into their unique ids.
    """
    dialog_ids = []
    for history, query, answer in dialogs:
        history = [[token_to_id.get(token, 'UNK') for token in sentence] for sentence in history]
        query = [token_to_id.get(token, 'UNK') for token in query]
        answer = [token_to_id.get(token, 'UNK') for token in answer]
        dialog_ids.append((history, query, answer))
    return dialog_ids


def get_tokenizer(dialogs, vocab_size=50000):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for history, query, answer in dialogs:
        tokens_all.extend([token for sentence in history for token in sentence] + query + answer)
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))

    count = [['UNK', -1], [PAD_TOKEN, -2]]
    count.extend(collections.Counter(vocab).most_common(vocab_size - 2))

    token_to_id = {token: i for i, (token, _) in enumerate(count)}
    return token_to_id


def pad_dialogs(dialogs, max_sentence_length, max_history_length, max_query_length, max_answer_length):
    """
    Pad sentences, dialogs, and queries to a consistence length.
    """
    for history, query, answer in dialogs:
        for sentence in history:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(PAD_ID)
            assert len(sentence) == max_sentence_length

        for _ in range(max_history_length - len(history)):
            history.append([PAD_ID for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(PAD_ID)

        for _ in range(max_answer_length - len(answer)):
            answer.append(PAD_ID)

        assert len(history) == max_history_length
        assert len(query) == max_query_length
        assert len(answer) == max_answer_length

    return dialogs


def truncate_dialogs(dialogs, max_length):
    dialogs_truncated = []
    for dialog, query, answer in dialogs:
        dialog_truncated = dialog[-max_length:]
        query_truncated = dialog[-max_length:]
        answer_truncated = dialog[-max_length:]
        dialogs_truncated.append((dialog_truncated, query_truncated, answer_truncated))
    return dialogs_truncated


def main():
    if not os.path.exists(FLAGS.dest_dir):
        os.makedirs(FLAGS.dest_dir)

    dialogs_path_train = os.path.join(FLAGS.source_dir, 'ubuntu/v2-trainset-filtered.csv')
    dialogs_path_test  = os.path.join(FLAGS.source_dir, 'ubuntu/v2-testset-filtered.csv')
    dataset_path_train = os.path.join(FLAGS.dest_dir, 'train.tfrecords')
    dataset_path_test  = os.path.join(FLAGS.dest_dir, 'test.tfrecords')
    metadata_path = os.path.join(FLAGS.dest_dir, 'meta.json')

    f_train = open(dialogs_path_train)
    f_test  = open(dialogs_path_test)

    dialogs_train = parse_dialogs(f_train.readlines())
    dialogs_test  = parse_dialogs(f_test.readlines())

    token_to_id = get_tokenizer(dialogs_train + dialogs_test)

    dialogs_token_train = tokenize_dialogs(dialogs_train, token_to_id)
    dialogs_token_test = tokenize_dialogs(dialogs_test, token_to_id)
    dialogs_token_all = dialogs_token_train + dialogs_token_test

    max_sentence_length = max([len(sentence) for dialog, _, _ in dialogs_token_all for sentence in dialog])
    max_dialog_length = max([len(dialog) for dialog, _, _ in dialogs_token_all])
    max_query_length = max([len(query) for _, query, _ in dialogs_token_all])
    max_answer_length = max([len(answer) for _, _, answer in dialogs_token_all])
    vocab_size = len(token_to_id)

    with open(metadata_path, 'w') as f:
        metadata = {
            'dataset_name': 'ubuntu',
            'dataset_train_size': len(dialogs_token_train),
            'dataset_test_size': len(dialogs_token_test),
            'max_sentence_length': max_sentence_length,
            'max_dialog_length': max_dialog_length,
            'max_query_length': max_query_length,
            'max_answer_length': max_answer_length,
            'vocab_size': vocab_size,
            'tokens': token_to_id,
            'datasets': {
                'train': os.path.basename(dataset_path_train),
                'test': os.path.basename(dataset_path_test),
            }
        }
        json.dump(metadata, f)

    dialogs_pad_train = pad_dialogs(dialogs_token_train, max_sentence_length, max_dialog_length, max_query_length, max_answer_length)
    dialogs_pad_test  = pad_dialogs(dialogs_token_test, max_sentence_length, max_dialog_length, max_query_length, max_answer_length)

    save_dataset(dialogs_pad_train, dataset_path_train)
    save_dataset(dialogs_pad_test, dataset_path_test)

if __name__ == '__main__':
    main()
