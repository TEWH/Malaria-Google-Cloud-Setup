WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

WARNING:tensorflow:From /srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py:109: The name tf.app.run is deprecated. Please use tf.compat.v1.app.run instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/config_util.py:102: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

W0318 02:12:50.941511 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/config_util.py:102: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:628: The name tf.logging.warning is deprecated. Please use tf.compat.v1.logging.warning instead.

W0318 02:12:50.947891 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:628: The name tf.logging.warning is deprecated. Please use tf.compat.v1.logging.warning instead.

WARNING:tensorflow:Forced number of epochs for all eval validations to be 1.
W0318 02:12:50.948068 140612716832576 model_lib.py:629] Forced number of epochs for all eval validations to be 1.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/config_util.py:488: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

W0318 02:12:50.948237 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/config_util.py:488: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

INFO:tensorflow:Maybe overwriting train_steps: 10000
I0318 02:12:50.948352 140612716832576 config_util.py:488] Maybe overwriting train_steps: 10000
INFO:tensorflow:Maybe overwriting use_bfloat16: False
I0318 02:12:50.948482 140612716832576 config_util.py:488] Maybe overwriting use_bfloat16: False
INFO:tensorflow:Maybe overwriting sample_1_of_n_eval_examples: 1
I0318 02:12:50.948570 140612716832576 config_util.py:488] Maybe overwriting sample_1_of_n_eval_examples: 1
INFO:tensorflow:Maybe overwriting eval_num_epochs: 1
I0318 02:12:50.948658 140612716832576 config_util.py:488] Maybe overwriting eval_num_epochs: 1
INFO:tensorflow:Maybe overwriting load_pretrained: True
I0318 02:12:50.948770 140612716832576 config_util.py:488] Maybe overwriting load_pretrained: True
INFO:tensorflow:Ignoring config override key: load_pretrained
I0318 02:12:50.948857 140612716832576 config_util.py:498] Ignoring config override key: load_pretrained
WARNING:tensorflow:Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
W0318 02:12:50.949949 140612716832576 model_lib.py:645] Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
INFO:tensorflow:create_estimator_and_inputs: use_tpu False, export_to_tpu False
I0318 02:12:50.950129 140612716832576 model_lib.py:680] create_estimator_and_inputs: use_tpu False, export_to_tpu False
INFO:tensorflow:Using config: {'_model_dir': '/srv/Malaria-Google-Cloud-Setup/Object-Detection/training', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fe28c1a9048>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
I0318 02:12:50.950772 140612716832576 estimator.py:212] Using config: {'_model_dir': '/srv/Malaria-Google-Cloud-Setup/Object-Detection/training', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fe28c1a9048>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
WARNING:tensorflow:Estimator's model_fn (<function create_model_fn.<locals>.model_fn at 0x7fe28c1a1620>) includes params argument, but params are not passed to Estimator.
W0318 02:12:50.951276 140612716832576 model_fn.py:630] Estimator's model_fn (<function create_model_fn.<locals>.model_fn at 0x7fe28c1a1620>) includes params argument, but params are not passed to Estimator.
INFO:tensorflow:Not using Distribute Coordinator.
I0318 02:12:50.952415 140612716832576 estimator_training.py:186] Not using Distribute Coordinator.
INFO:tensorflow:Running training and evaluation locally (non-distributed).
I0318 02:12:50.952702 140612716832576 training.py:612] Running training and evaluation locally (non-distributed).
INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps None or save_checkpoints_secs 600.
I0318 02:12:50.953084 140612716832576 training.py:700] Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps None or save_checkpoints_secs 600.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/training_util.py:236: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
W0318 02:12:50.964607 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/training_util.py:236: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/data_decoders/tf_example_decoder.py:182: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.

W0318 02:12:50.980277 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/data_decoders/tf_example_decoder.py:182: The name tf.FixedLenFeature is deprecated. Please use tf.io.FixedLenFeature instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/data_decoders/tf_example_decoder.py:197: The name tf.VarLenFeature is deprecated. Please use tf.io.VarLenFeature instead.

W0318 02:12:50.980576 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/data_decoders/tf_example_decoder.py:197: The name tf.VarLenFeature is deprecated. Please use tf.io.VarLenFeature instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:64: The name tf.gfile.Glob is deprecated. Please use tf.io.gfile.glob instead.

W0318 02:12:50.995173 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:64: The name tf.gfile.Glob is deprecated. Please use tf.io.gfile.glob instead.

WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
W0318 02:12:50.996431 140612716832576 dataset_builder.py:72] num_readers has been reduced to 1 to match input file shards.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:86: parallel_interleave (from tensorflow.contrib.data.python.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.experimental.parallel_interleave(...)`.
W0318 02:12:51.003051 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:86: parallel_interleave (from tensorflow.contrib.data.python.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.experimental.parallel_interleave(...)`.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/data/python/ops/interleave_ops.py:77: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_determinstic`.
W0318 02:12:51.003234 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/data/python/ops/interleave_ops.py:77: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_determinstic`.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:155: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
W0318 02:12:51.032860 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:155: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.logging.warn is deprecated. Please use tf.compat.v1.logging.warn instead.

W0318 02:12:53.073490 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.logging.warn is deprecated. Please use tf.compat.v1.logging.warn instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.is_nan is deprecated. Please use tf.math.is_nan instead.

W0318 02:13:05.133908 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.is_nan is deprecated. Please use tf.math.is_nan instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/ops.py:493: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
W0318 02:13:05.273462 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/ops.py:493: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

W0318 02:13:08.772317 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/operators/control_flow.py:1004: sample_distorted_bounding_box (from tensorflow.python.ops.image_ops_impl) is deprecated and will be removed in a future version.
Instructions for updating:
`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.
W0318 02:13:14.725119 140612716832576 api.py:332] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/operators/control_flow.py:1004: sample_distorted_bounding_box (from tensorflow.python.ops.image_ops_impl) is deprecated and will be removed in a future version.
Instructions for updating:
`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.image.resize_images is deprecated. Please use tf.image.resize instead.

W0318 02:13:22.754239 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.image.resize_images is deprecated. Please use tf.image.resize instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:166: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
W0318 02:13:23.763992 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/inputs.py:166: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.string_to_hash_bucket_fast is deprecated. Please use tf.strings.to_hash_bucket_fast instead.

W0318 02:13:26.271663 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.string_to_hash_bucket_fast is deprecated. Please use tf.strings.to_hash_bucket_fast instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:158: batch_and_drop_remainder (from tensorflow.contrib.data.python.ops.batching) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.batch(..., drop_remainder=True)`.
W0318 02:13:27.053746 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/dataset_builder.py:158: batch_and_drop_remainder (from tensorflow.contrib.data.python.ops.batching) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.batch(..., drop_remainder=True)`.
INFO:tensorflow:Calling model_fn.
I0318 02:13:27.071187 140612716832576 estimator.py:1148] Calling model_fn.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:589: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.

W0318 02:13:27.434641 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:589: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:597: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

W0318 02:13:27.434905 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:597: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/layers/python/layers/layers.py:1057: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
W0318 02:13:27.448878 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/contrib/layers/python/layers/layers.py:1057: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/core/anchor_generator.py:171: The name tf.assert_equal is deprecated. Please use tf.compat.v1.assert_equal instead.

W0318 02:13:30.599747 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/core/anchor_generator.py:171: The name tf.assert_equal is deprecated. Please use tf.compat.v1.assert_equal instead.

INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.610563 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.646003 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.681240 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.715613 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.751131 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:13:30.786845 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/variables_helper.py:179: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

W0318 02:13:30.828049 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/variables_helper.py:179: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/variables_helper.py:139: The name tf.train.NewCheckpointReader is deprecated. Please use tf.compat.v1.train.NewCheckpointReader instead.

W0318 02:13:30.829385 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/variables_helper.py:139: The name tf.train.NewCheckpointReader is deprecated. Please use tf.compat.v1.train.NewCheckpointReader instead.

W0318 02:13:30.834344 140612716832576 variables_helper.py:154] Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_512/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 256, 512]], model variable shape: [[3, 3, 256, 512]]. This variable will not be initialized from the checkpoint.
W0318 02:13:30.834544 140612716832576 variables_helper.py:154] Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_3_3x3_s2_256/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 128, 256]], model variable shape: [[3, 3, 128, 256]]. This variable will not be initialized from the checkpoint.
W0318 02:13:30.834677 140612716832576 variables_helper.py:154] Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_4_3x3_s2_256/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 128, 256]], model variable shape: [[3, 3, 128, 256]]. This variable will not be initialized from the checkpoint.
W0318 02:13:30.834796 140612716832576 variables_helper.py:154] Variable [FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/weights] is available in checkpoint, but has an incompatible shape with model variable. Checkpoint shape: [[1, 1, 64, 128]], model variable shape: [[3, 3, 64, 128]]. This variable will not be initialized from the checkpoint.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:353: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.

W0318 02:13:30.834963 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:353: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/box_coders/faster_rcnn_box_coder.py:82: The name tf.log is deprecated. Please use tf.math.log instead.

W0318 02:13:31.979682 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/box_coders/faster_rcnn_box_coder.py:82: The name tf.log is deprecated. Please use tf.math.log instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:1163: The name tf.summary.scalar is deprecated. Please use tf.compat.v1.summary.scalar instead.

W0318 02:13:36.376604 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:1163: The name tf.summary.scalar is deprecated. Please use tf.compat.v1.summary.scalar instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/core/losses.py:177: The name tf.losses.huber_loss is deprecated. Please use tf.compat.v1.losses.huber_loss instead.

W0318 02:13:36.383455 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/core/losses.py:177: The name tf.losses.huber_loss is deprecated. Please use tf.compat.v1.losses.huber_loss instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/core/losses.py:183: The name tf.losses.Reduction is deprecated. Please use tf.compat.v1.losses.Reduction instead.

W0318 02:13:36.384834 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/core/losses.py:183: The name tf.losses.Reduction is deprecated. Please use tf.compat.v1.losses.Reduction instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:1275: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.

W0318 02:13:37.044275 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/meta_architectures/ssd_meta_arch.py:1275: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:380: The name tf.train.get_or_create_global_step is deprecated. Please use tf.compat.v1.train.get_or_create_global_step instead.

W0318 02:13:37.047393 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:380: The name tf.train.get_or_create_global_step is deprecated. Please use tf.compat.v1.train.get_or_create_global_step instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/learning_schedules.py:66: The name tf.train.exponential_decay is deprecated. Please use tf.compat.v1.train.exponential_decay instead.

W0318 02:13:37.047721 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/learning_schedules.py:66: The name tf.train.exponential_decay is deprecated. Please use tf.compat.v1.train.exponential_decay instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/builders/optimizer_builder.py:47: The name tf.train.RMSPropOptimizer is deprecated. Please use tf.compat.v1.train.RMSPropOptimizer instead.

W0318 02:13:37.056534 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/builders/optimizer_builder.py:47: The name tf.train.RMSPropOptimizer is deprecated. Please use tf.compat.v1.train.RMSPropOptimizer instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:398: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

W0318 02:13:37.056757 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:398: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/rmsprop.py:119: calling Ones.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
W0318 02:13:39.727293 140612716832576 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/rmsprop.py:119: calling Ones.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:515: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

W0318 02:13:46.036462 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:515: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:519: The name tf.add_to_collection is deprecated. Please use tf.compat.v1.add_to_collection instead.

W0318 02:13:47.007514 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:519: The name tf.add_to_collection is deprecated. Please use tf.compat.v1.add_to_collection instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:520: The name tf.train.Scaffold is deprecated. Please use tf.compat.v1.train.Scaffold instead.

W0318 02:13:47.007813 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:520: The name tf.train.Scaffold is deprecated. Please use tf.compat.v1.train.Scaffold instead.

INFO:tensorflow:Done calling model_fn.
I0318 02:13:47.008336 140612716832576 estimator.py:1150] Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
I0318 02:13:47.009804 140612716832576 basic_session_run_hooks.py:541] Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
I0318 02:13:51.705414 140612716832576 monitored_session.py:240] Graph was finalized.
2020-03-18 02:13:51.705890: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-18 02:13:51.712670: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2020-03-18 02:13:51.713007: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x567ef70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-18 02:13:51.713044: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-18 02:13:51.715847: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-18 02:13:51.821383: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-18 02:13:51.821993: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x54900e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-18 02:13:51.822036: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
2020-03-18 02:13:51.822279: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-18 02:13:51.822745: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
pciBusID: 0000:00:04.0
2020-03-18 02:13:51.822957: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.823089: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10.0'; dlerror: libcublas.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.823206: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10.0'; dlerror: libcufft.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.823316: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10.0'; dlerror: libcurand.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.823453: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10.0'; dlerror: libcusolver.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.823567: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10.0'; dlerror: libcusparse.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:13:51.827689: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-18 02:13:51.827736: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1641] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-18 02:13:51.827764: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-18 02:13:51.827784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-18 02:13:51.827798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
INFO:tensorflow:Restoring parameters from /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt-0
I0318 02:13:51.829767 140612716832576 saver.py:1284] Restoring parameters from /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt-0
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/saver.py:1069: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file utilities to get mtimes.
W0318 02:13:54.312800 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/saver.py:1069: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
Instructions for updating:
Use standard file utilities to get mtimes.
INFO:tensorflow:Running local_init_op.
I0318 02:13:55.878977 140612716832576 session_manager.py:500] Running local_init_op.
INFO:tensorflow:Done running local_init_op.
I0318 02:13:56.482404 140612716832576 session_manager.py:502] Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt.
I0318 02:14:13.580572 140612716832576 basic_session_run_hooks.py:606] Saving checkpoints for 0 into /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt.
INFO:tensorflow:loss = 19.999062, step = 1
I0318 02:14:47.739792 140612716832576 basic_session_run_hooks.py:262] loss = 19.999062, step = 1
INFO:tensorflow:Saving checkpoints for 66 into /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt.
I0318 02:24:18.939504 140612716832576 basic_session_run_hooks.py:606] Saving checkpoints for 66 into /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt.
INFO:tensorflow:Calling model_fn.
I0318 02:24:22.135867 140612716832576 estimator.py:1148] Calling model_fn.
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.774742 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.812345 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.850819 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.889180 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.925652 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0318 02:24:24.962163 140612716832576 convolutional_box_predictor.py:151] depth of additional conv before box predictor: 0
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:796: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
W0318 02:24:25.893966 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py:796: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:498: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
W0318 02:24:26.132348 140612716832576 deprecation.py:323] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:498: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:1044: The name tf.summary.image is deprecated. Please use tf.compat.v1.summary.image instead.

W0318 02:24:26.323249 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/utils/visualization_utils.py:1044: The name tf.summary.image is deprecated. Please use tf.compat.v1.summary.image instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:484: The name tf.metrics.mean is deprecated. Please use tf.compat.v1.metrics.mean instead.

W0318 02:24:26.419739 140612716832576 module_wrapper.py:139] From /usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py:484: The name tf.metrics.mean is deprecated. Please use tf.compat.v1.metrics.mean instead.

INFO:tensorflow:Done calling model_fn.
I0318 02:24:26.760924 140612716832576 estimator.py:1150] Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2020-03-18T02:24:26Z
I0318 02:24:26.781388 140612716832576 evaluation.py:255] Starting evaluation at 2020-03-18T02:24:26Z
INFO:tensorflow:Graph was finalized.
I0318 02:24:27.281028 140612716832576 monitored_session.py:240] Graph was finalized.
2020-03-18 02:24:27.282322: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-18 02:24:27.282859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla V100-SXM2-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.53
pciBusID: 0000:00:04.0
2020-03-18 02:24:27.283170: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283331: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10.0'; dlerror: libcublas.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283444: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10.0'; dlerror: libcufft.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283667: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10.0'; dlerror: libcurand.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283830: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10.0'; dlerror: libcusolver.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283943: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10.0'; dlerror: libcusparse.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2020-03-18 02:24:27.283984: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-18 02:24:27.284004: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1641] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-18 02:24:27.284034: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-18 02:24:27.284094: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-18 02:24:27.284106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
INFO:tensorflow:Restoring parameters from /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt-66
I0318 02:24:27.285320 140612716832576 saver.py:1284] Restoring parameters from /srv/Malaria-Google-Cloud-Setup/Object-Detection/training/model.ckpt-66
INFO:tensorflow:Running local_init_op.
I0318 02:24:28.442014 140612716832576 session_manager.py:500] Running local_init_op.
INFO:tensorflow:Done running local_init_op.
I0318 02:24:28.598284 140612716832576 session_manager.py:502] Done running local_init_op.
INFO:tensorflow:Performing evaluation on 120 images.
I0318 02:25:59.939740 140609729844992 coco_evaluation.py:205] Performing evaluation on 120 images.
creating index...
index created!
INFO:tensorflow:Loading and preparing annotation results...
I0318 02:25:59.944373 140609729844992 coco_tools.py:115] Loading and preparing annotation results...
INFO:tensorflow:DONE (t=0.05s)
I0318 02:25:59.995872 140609729844992 coco_tools.py:137] DONE (t=0.05s)
creating index...
index created!
2020-03-18 02:26:00.089128: W tensorflow/core/framework/op_kernel.cc:1639] Invalid argument: TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.
Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 117, in linspace
    num = operator.index(num)

TypeError: 'numpy.float64' object cannot be interpreted as an integer


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 235, in __call__
    ret = func(*args)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 384, in first_value_func
    self._metrics = self.evaluate()

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 215, in evaluate
    coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_tools.py", line 176, in __init__
    iouType=iou_type)

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 76, in __init__
    self.params = Params(iouType=iouType) # parameters

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 527, in __init__
    self.setDetParams()

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 507, in setDetParams
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

  File "<__array_function__ internals>", line 6, in linspace

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 121, in linspace
    .format(type(num)))

TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.


Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.OutOfRangeError: End of sequence
	 [[{{node IteratorGetNext}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/evaluation.py", line 272, in _evaluate_once
    session.run(eval_ops, feed_dict)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 754, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1259, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1360, in run
    raise six.reraise(*original_exc_info)
  File "/usr/local/lib/python3.6/dist-packages/six.py", line 696, in reraise
    raise value
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1345, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1418, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1176, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.OutOfRangeError: End of sequence
	 [[node IteratorGetNext (defined at usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'IteratorGetNext':
  File "srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 109, in <module>
    tf.app.run()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "usr/local/lib/python3.6/dist-packages/absl/app.py", line 299, in run
    _run_main(main, args)
  File "usr/local/lib/python3.6/dist-packages/absl/app.py", line 250, in _run_main
    sys.exit(main(argv))
  File "srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 105, in main
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 473, in train_and_evaluate
    return executor.run()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 613, in run
    return self.run_local()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 714, in run_local
    saving_listeners=saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
    saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1494, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 754, in run
    run_metadata=run_metadata)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1259, in run
    run_metadata=run_metadata)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1345, in run
    return self._sess.run(*args, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1426, in run
    run_metadata=run_metadata))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 594, in after_run
    if self._save(run_context.session, global_step):
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 619, in _save
    if l.after_save(session, step):
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 519, in after_save
    self._evaluate(global_step_value)  # updates self.eval_result
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 539, in _evaluate
    self._evaluator.evaluate_and_export())
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 920, in evaluate_and_export
    hooks=self._eval_spec.hooks)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 480, in evaluate
    name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 522, in _actual_eval
    return _evaluate()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 504, in _evaluate
    self._evaluate_build_graph(input_fn, hooks, checkpoint_path))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1511, in _evaluate_build_graph
    self._call_model_fn_eval(input_fn, self.config))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1544, in _call_model_fn_eval
    input_fn, ModeKeys.EVAL)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1025, in _get_features_and_labels_from_input_fn
    self._call_input_fn(input_fn, mode))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/util.py", line 65, in parse_input_fn_result
    result = iterator.get_next()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/data/ops/iterator_ops.py", line 426, in get_next
    name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/gen_dataset_ops.py", line 2518, in iterator_get_next
    output_shapes=output_shapes, name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InvalidArgumentError: TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.
Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 117, in linspace
    num = operator.index(num)

TypeError: 'numpy.float64' object cannot be interpreted as an integer


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 235, in __call__
    ret = func(*args)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 384, in first_value_func
    self._metrics = self.evaluate()

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 215, in evaluate
    coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_tools.py", line 176, in __init__
    iouType=iou_type)

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 76, in __init__
    self.params = Params(iouType=iouType) # parameters

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 527, in __init__
    self.setDetParams()

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 507, in setDetParams
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

  File "<__array_function__ internals>", line 6, in linspace

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 121, in linspace
    .format(type(num)))

TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.


	 [[{{node PyFunc_3}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 109, in <module>
    tf.app.run()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 299, in run
    _run_main(main, args)
  File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 250, in _run_main
    sys.exit(main(argv))
  File "/srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 105, in main
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 473, in train_and_evaluate
    return executor.run()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 613, in run
    return self.run_local()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 714, in run_local
    saving_listeners=saving_listeners)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
    saving_listeners)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1494, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 754, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1259, in run
    run_metadata=run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1360, in run
    raise six.reraise(*original_exc_info)
  File "/usr/local/lib/python3.6/dist-packages/six.py", line 696, in reraise
    raise value
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1345, in run
    return self._sess.run(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1426, in run
    run_metadata=run_metadata))
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 594, in after_run
    if self._save(run_context.session, global_step):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 619, in _save
    if l.after_save(session, step):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 519, in after_save
    self._evaluate(global_step_value)  # updates self.eval_result
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 539, in _evaluate
    self._evaluator.evaluate_and_export())
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 920, in evaluate_and_export
    hooks=self._eval_spec.hooks)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 480, in evaluate
    name=name)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 522, in _actual_eval
    return _evaluate()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 511, in _evaluate
    output_dir=self.eval_dir(name))
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1619, in _evaluate_run
    config=self._session_config)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/evaluation.py", line 272, in _evaluate_once
    session.run(eval_ops, feed_dict)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 861, in __exit__
    self._close_internal(exception_type)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 894, in _close_internal
    h.end(self._coordinated_creator.tf_sess)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 951, in end
    self._final_ops, feed_dict=self._final_ops_feed_dict)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.
Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 117, in linspace
    num = operator.index(num)

TypeError: 'numpy.float64' object cannot be interpreted as an integer


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 235, in __call__
    ret = func(*args)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 384, in first_value_func
    self._metrics = self.evaluate()

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 215, in evaluate
    coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)

  File "/usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_tools.py", line 176, in __init__
    iouType=iou_type)

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 76, in __init__
    self.params = Params(iouType=iouType) # parameters

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 527, in __init__
    self.setDetParams()

  File "/usr/local/lib/python3.6/dist-packages/pycocotools/cocoeval.py", line 507, in setDetParams
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)

  File "<__array_function__ internals>", line 6, in linspace

  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 121, in linspace
    .format(type(num)))

TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.


	 [[node PyFunc_3 (defined at usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'PyFunc_3':
  File "srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 109, in <module>
    tf.app.run()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "usr/local/lib/python3.6/dist-packages/absl/app.py", line 299, in run
    _run_main(main, args)
  File "usr/local/lib/python3.6/dist-packages/absl/app.py", line 250, in _run_main
    sys.exit(main(argv))
  File "srv/Malaria-Google-Cloud-Setup/Object-Detection/models/research/object_detection/model_main.py", line 105, in main
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 473, in train_and_evaluate
    return executor.run()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 613, in run
    return self.run_local()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 714, in run_local
    saving_listeners=saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
    saving_listeners)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1494, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 754, in run
    run_metadata=run_metadata)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1259, in run
    run_metadata=run_metadata)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1345, in run
    return self._sess.run(*args, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/monitored_session.py", line 1426, in run
    run_metadata=run_metadata))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 594, in after_run
    if self._save(run_context.session, global_step):
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/training/basic_session_run_hooks.py", line 619, in _save
    if l.after_save(session, step):
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 519, in after_save
    self._evaluate(global_step_value)  # updates self.eval_result
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 539, in _evaluate
    self._evaluator.evaluate_and_export())
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/training.py", line 920, in evaluate_and_export
    hooks=self._eval_spec.hooks)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 480, in evaluate
    name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 522, in _actual_eval
    return _evaluate()
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 504, in _evaluate
    self._evaluate_build_graph(input_fn, hooks, checkpoint_path))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1511, in _evaluate_build_graph
    self._call_model_fn_eval(input_fn, self.config))
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1547, in _call_model_fn_eval
    features, labels, ModeKeys.EVAL, config)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py", line 1149, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/object_detection/model_lib.py", line 482, in model_fn
    eval_config, list(category_index.values()), eval_dict)
  File "usr/local/lib/python3.6/dist-packages/object_detection/eval_util.py", line 947, in get_eval_metric_ops_for_evaluators
    eval_dict))
  File "usr/local/lib/python3.6/dist-packages/object_detection/metrics/coco_evaluation.py", line 394, in get_estimator_eval_metric_ops
    first_value_op = tf.py_func(first_value_func, [], tf.float32)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/util/deprecation.py", line 324, in new_func
    return func(*args, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 513, in py_func
    return py_func_common(func, inp, Tout, stateful, name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 495, in py_func_common
    func=func, inp=inp, Tout=Tout, stateful=stateful, eager=False, name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/script_ops.py", line 318, in _internal_py_func
    input=inp, token=token, Tout=Tout, name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/gen_script_ops.py", line 170, in py_func
    "PyFunc", input=input, token=token, Tout=Tout, name=name)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "usr/local/lib/python3.6/dist-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()
