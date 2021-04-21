# -*- coding: UTF-8 -*-

import os
import time
import argparse
import math
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from datetime import datetime

from tensorflow.python.platform import gfile


partition_num = 1
tensor_item_key = ["embedding_weights"]

def create_variable(tensor_name, tensor_value, dtype):
    if partition_num > 1:
        partitioner = tf.fixed_size_partitioner(partition_num, axis=0),
        return tf.get_variable(name=tensor_name, initializer=tensor_value, dtype=dtype,
                               partitioner=partitioner, validate_shape=False)
    else:
        return tf.get_variable(name=tensor_name, initializer=tensor_value, 
                               dtype=dtype, validate_shape=False)


def create_new_variable(tensor_name, shape, dtype):
    init = tf.compat.v1.constant_initializer(0)
    if partition_num > 1:
        partitioner = tf.fixed_size_partitioner(partition_num, axis=0),
        return tf.get_variable(name=tensor_name, shape=shape, dtype=dtype, 
                               initializer=init, partitioner=partitioner)
    else:
        return tf.get_variable(name=tensor_name, shape=shape, 
                               dtype=dtype, initializer=init)


def handle_extract_feature_embedding(feature, embedding_var):

    embedding_size = embedding_var.shape[0]
    embedding_dim = embedding_var.shape[1]

    embedding_l1_norm = np.linalg.norm(embedding_var, axis=1, ord=1)
    non_zero_index = np.nonzero(embedding_l1_norm)[0]
    non_zero_embedding = np.take(embedding_var, non_zero_index, axis=0)
    total_non_zero_num = non_zero_index.shape[0]

    zero_embedding = np.zeros(shape=(embedding_dim,), dtype=np.float)
    new_embedding_var = np.insert(non_zero_embedding, obj=0, values=zero_embedding, axis=0)

    non_zero_range = np.arange(1, total_non_zero_num+1)
    new_embedding_map_array = np.zeros(shape=(embedding_size,), dtype=np.int64)
    new_embedding_map_array[non_zero_index] = non_zero_range

    print("Feature {} total non zero embedding num: {}/{}".format(feature, total_non_zero_num, embedding_size))

    return new_embedding_var, new_embedding_map_array

def get_embedding_feature_name(feature_info):
    return feature_info["embedding_weights"]

def get_embedding_map_name(feature_info):
    name = "{}_maps".format(feature_info["embedding_weights"])
    return name

def get_embedding_feautre_var(feature_info, reader):
    name = get_embedding_feature_name(feature_info)
    embedding_var = reader.get_tensor(name)
    return name, embedding_var


def extract_one_feature_embedding(feature, feature_info, reader, sess, 
                       name_dtype_map, name_shape_map):
    print("=== Prepare extract feature embedding: {} ===".format(feature))

    def tf_extract_op(feature):
        print("=== Start feautre embedding extract: {} ===".format(feature))
        feature = feature.decode('ascii')
        embedding_name, embedding_var = get_embedding_feautre_var(feature_info, reader)

        new_embedding_var, embedding_map_var = \
            handle_extract_feature_embedding(feature, embedding_var)

        print("=== Finish feautre embedding extract: {} ===".format(feature))
        return new_embedding_var, embedding_map_var

    feature_input = tf.compat.v1.placeholder(tf.string)
    input_list = [feature_input]
 
    embedding_var_name = get_embedding_feature_name(feature_info)
    embedding_type = name_dtype_map[embedding_var_name]

    return_list = [embedding_type, tf.int64]

    new_embedding_var, embedding_map_var = tf.compat.v1.py_func(tf_extract_op, inp=input_list, Tout=return_list)

    embedding_map_shape = embedding_map_var.shape
    fead_dict = {feature_input:feature}

    with tf.control_dependencies([new_embedding_var, embedding_map_var]):
        new_embedding_var = tf.convert_to_tensor(new_embedding_var, dtype=embedding_type)
        var1 = create_variable(embedding_var_name, new_embedding_var, dtype=embedding_type)

        map_var = tf.convert_to_tensor(embedding_map_var, dtype=tf.int64)
        embedding_map_var_name = get_embedding_map_name(feature_info)
        var2 = create_variable(embedding_map_var_name, map_var, dtype=tf.int64)

        print("Feature embedding var: {}".format(embedding_var_name))
        print("Feature embedding map: {}".format(embedding_map_var_name))
    
    print("=== Done prepare extract feautre embedding: {} ===".format(feature))
    return fead_dict, set([embedding_var_name]), {feature: embedding_map_var_name}


def handle_unchanged_feature(var_name, reader, sess, name_dtype_map):

    def tf_convert_unchagned_op(var_name):
        return reader.get_tensor(var_name)

    var_input = tf.compat.v1.placeholder(tf.string)
    var_type = name_dtype_map[var_name]
    new_var = tf.compat.v1.py_func(tf_convert_unchagned_op, [var_input], var_type)

    fead_dict = {var_input: var_name}
    with tf.control_dependencies([new_var]):
        var = create_variable(var_name, new_var, dtype=var_type)

    #print("Not changed var: {}".format(var_name))
    return fead_dict


def write_checkpoint_file(output_dir, latest_ckp):
    # eg. latest_ckp: /root/tensorflow/id_table/checkpoint-junqin-wd-test/model.ckpt-42532
    checkpoint_file = os.path.join(output_dir, "checkpoint")
    model_ckpt = latest_ckp.split("/")[-1]
    with tf.gfile.GFile(checkpoint_file, 'w') as f:
        f.write('model_checkpoint_path: "{}"\n'.format(model_ckpt))
        f.write('all_model_checkpoint_paths: "{}"'.format(model_ckpt))


def check_feature_variable(feature_info, name_shape_map):
    var_name = feature_info["embedding_weights"]
    if var_name not in name_shape_map:
        print("Error: {} not in checkpoint".format(var_name))
        exit(1)


def extract_gftrl_embeddings(latest_ckp, output_dir, feature_spec):
    with tf.Session() as sess:
        reader = tf.train.NewCheckpointReader(latest_ckp)
        name_shape_map = reader.get_variable_to_shape_map()
        name_dtype_map = reader.get_variable_to_dtype_map()

        var_changed = set()
        feed_dict = dict()
        feature_embedding_map_name = dict()
        for feature in feature_spec:
            check_feature_variable(feature_spec[feature], name_shape_map)

        for feature in feature_spec:
            feature_info = feature_spec[feature]
            feature_feed, var_names, embedding_map = \
                extract_one_feature_embedding(feature, feature_info, reader, sess,
			                      name_dtype_map, name_shape_map)
            feed_dict.update(feature_feed)
            feature_embedding_map_name.update(embedding_map)
            var_changed = var_changed | var_names

        for var_name in name_shape_map:
            if var_name in var_changed:
                continue
            if var_name.find('GFtrl') >=0:
               continue
            if var_name.find('Ftrl') >=0:
               continue
            if var_name.find('Adagrad') >=0:
               continue
            this_feed_dict = handle_unchanged_feature(var_name, reader, 
                                                      sess, name_dtype_map)
            feed_dict.update(this_feed_dict)

        new_ckpt = os.path.join(output_dir, latest_ckp.split('/')[-1])
        sess.run(tf.global_variables_initializer(), feed_dict)
        saver = tf.train.Saver(sharded=True)
        saver.save(sess, new_ckpt, write_meta_graph=False)
        write_checkpoint_file(output_dir, latest_ckp)
        print("Saved new ckpt to: {}".format(new_ckpt))
   
        reader = tf.train.NewCheckpointReader(new_ckpt)
        name_shape_map = reader.get_variable_to_shape_map()
        for feature in feature_spec:
            embedding_feature = get_embedding_feature_name(feature_spec[feature])
            feature_spec[feature]['new_shape'] = name_shape_map[embedding_feature]
        new_feature_spec_file = os.path.join(output_dir, 'feature_spec.json')
        with tf.gfile.GFile(new_feature_spec_file, 'w') as f:
            f.write(json.dumps(feature_spec))


def check_feature_with_file(feature):
    feature_with_file = feature.split(':')
    if len(feature_with_file) != 2:
        print("Error: feature should in <feature name>:<file path> format: {}".format(feature))
        exit(1)
    if not feature_with_file[0] or not feature_with_file[1]:
        print("Error: feature should in <feature name>:<file path> format: {}".format(feature))
        exit(1)
    feature_name, file_path = feature_with_file
    if gfile.IsDirectory(file_path):
        print("Error: feature id file path should be a file: {}".format(feature))
        exit(1)
    if not gfile.Exists(file_path):
        print("Error: feature id file path not exists: {}".format(feature))
        exit(1)
    return feature_name, file_path


def check_file_exist(file_path):
    if gfile.IsDirectory(file_path):
        print("Error: file path is not a file: {}".format(file_path))
        exit(1)
    if not gfile.Exists(file_path):
        print("Error: file path not exists: {}".format(file_path))
        exit(1)
    return True


def get_feature_spec(spec_file):
    #{
    #  "feature1": {
    #    "embedding_weights": "dnn/input_from_feature_columns/input_layer/feature1_embedding/embedding_weights",
    #  }
    #}
    with tf.gfile.GFile(spec_file, 'r') as f:
        feature_spec = f.read()

    if not feature_spec:
        print("Error: empty spec file: {}".format(spec_file))
        exit(1)

    try:
        feature_spec = json.loads(feature_spec)
    except Exception as e:
        print("Error: feature spec not in Json format: {}".format(e))
        exit(1)

    for feature in feature_spec:
        feature_info = feature_spec[feature]
        for i in tensor_item_key:
            if i not in feature_info:
                print("Error: item {} not in feature {} spec".format(i, feature))
                exit(1)

    return feature_spec


def extract_vars_from_saved_model(savedmodel_dir, output_dir, savedmodel_tag):
    print("=== Start extract vars from saved model ===")
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        MetaGraphDef = tf.saved_model.loader.load(sess, tags=[savedmodel_tag], export_dir=savedmodel_dir)
        #graph = tf.get_default_graph()

        global_step = sess.graph.get_tensor_by_name("global_step:0")
        global_step_value = sess.run(global_step)

        new_ckpt = os.path.join(output_dir, 'model.ckpt-{}'.format(global_step_value))
        saver = tf.train.Saver()
        saver.save(sess, new_ckpt, write_meta_graph=False)
        write_checkpoint_file(output_dir, new_ckpt)
        print("Extract vars from saved model to ckpt: {}".format(new_ckpt))
        return output_dir


def check_args(args):
    if args.checkpoint_dir and args.savedmodel_dir:
        print("Error: Only could pass one of input-savedmodel-dir and input-checkpoint-dir")
        exit(1)

    if args.checkpoint_dir and not gfile.IsDirectory(args.checkpoint_dir):
        print("Error: Checkpoint dir not exist: {}".format(args.checkpoint_dir))
        exit(1)

    if args.savedmodel_dir and not gfile.IsDirectory(args.savedmodel_dir):
        print("Error: Saved model dir dir not exist: {}".format(args.savedmodel_dir))
        exit(1)

    if not gfile.IsDirectory(args.output_dir):
        print("Error: Output dir not exist: {}".format(args.output_dir))
        exit(1)


def main(args):
    check_args(args)

    check_file_exist(args.feature_spec_file)

    if args.savedmodel_dir:
        checkpoint_dir = extract_vars_from_saved_model(args.savedmodel_dir, 
                                                       args.output_dir,
                                                       args.savedmodel_tag)
    else:
        checkpoint_dir = args.checkpoint_dir

    latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
    print("Latest checkpoint: {}".format(latest_ckp))
    if not latest_ckp:
        print("Error: There is no checkpoint in the dir: {}".format(args.checkpoint_dir))
        exit(1)

    feature_spec = get_feature_spec(args.feature_spec_file)
    extract_gftrl_embeddings(latest_ckp, args.output_dir, feature_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Opal tools to remove zero embedding vectors')
    parser.add_argument('--input-savedmodel-dir', dest="savedmodel_dir", type=str,
        required=False, help='Saved model input path')
    parser.add_argument('--input-checkpoint-dir', dest="checkpoint_dir", type=str,
        required=False, help='Checkpoint input path')
    parser.add_argument("--savedmodel-tag", dest="savedmodel_tag",
                        type=str, default=tf.saved_model.tag_constants.SERVING,
                        help="tag get by 'saved_model_cli show'")
    parser.add_argument('--output-checkpoint-dir', dest="output_dir", type=str,
        required=True, help='Checkpoint output path')
    parser.add_argument("--feature-json-spec-file", dest="feature_spec_file", type=str,
        required=True, help="feature spec detail file in Json format")
    args = parser.parse_args()
    main(args)

