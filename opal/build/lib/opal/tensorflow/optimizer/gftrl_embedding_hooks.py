import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=g-bad-import-order

tf.disable_v2_behavior()

"""
Usage example:
  name_list = ["dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights"]
  eval_hooks = [get_zero_embedding_check_hook(name_list)]
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn(flags_obj), steps=200, hooks=eval_hooks)
"""
def get_zero_embedding_check_hook(embedding_name_list):
  if not isinstance(embedding_name_list, list):
    raise Exception("Error: embedding name list {} is not a list".format(embedding_name_list))
  if not embedding_name_list:
    raise Exception("Error: embedding name list {} is empty".format(embedding_name_list))

  class ZeroEmbeddingCheckHook(tf.estimator.SessionRunHook):

    def end(self, session):
      tf.logging.info("=== Opal ZeroEmbeddingCheckHook begin ===")
      all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      weigth_map_dict = dict()
      for variable in all_variables:
        for embedding in embedding_name_list:
          if variable.name.find(embedding) == 0:
            var_list = weigth_map_dict.get(embedding, [])
            var_list.append(variable)
            weigth_map_dict[embedding] = var_list

      tf.logging.info(weigth_map_dict)
      for embedding in weigth_map_dict:
        total_num = 0
        total_non_zero = 0
        for var in weigth_map_dict[embedding]:
            embedding_var = session.run(var)
            embedding_l1_norm = np.linalg.norm(embedding_var, axis=1, ord=1)
            total_non_zero += np.count_nonzero(embedding_l1_norm)
            total_num += embedding_var.shape[0]

        ratio = "{0:.1%}".format(total_non_zero/float(total_num))
        tf.logging.info("Embedding {} Non-zero size: {}/{}, Ratio: {}".format(embedding, total_non_zero, total_num, ratio))
      tf.logging.info("=== Opal ZeroEmbeddingCheckHook end ===")

  return ZeroEmbeddingCheckHook()

