# -*- coding: utf-8 -*-
import logging
import os
import pprint as pp
from collections import deque
from datetime import datetime
from imp import reload

import numpy as np
import tensorflow as tf

from utils import shape
import pandas as pd


class TFBaseModel(object):
    """Interface containing some boilerplate code for training tensorflow models.

    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.

    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(
            self,
            reader,
            batch_size=128,
            num_training_steps=20000,
            learning_rate=.01,
            optimizer='adam',

            grad_clip=5,
            regularization_constant=0.0,
            keep_prob=1.0,
            early_stopping_steps=3000,
            warm_start_init_step=0,

            num_restarts=None,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=100,
            log_interval=20,
            loss_averaging_window=100,  # 没有传入该参数

            num_validation_batches=1,
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions'
    ):

        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        # 梯度裁剪
        self.grad_clip = grad_clip
        # 参数L2正则化
        self.regularization_constant = regularization_constant
        # Dropout 比例
        self.keep_prob_scalar = keep_prob
        # 没有权重更新多少步后停止训练
        self.early_stopping_steps = early_stopping_steps
        # 1：基于已有权重训练 0:从头开始训练
        self.warm_start_init_step = warm_start_init_step
        # None:早停条件满足就停止；早停条件满足、且重启次数超过该阈值才停止
        self.num_restarts = num_restarts
        # 权重滑动平均
        self.enable_parameter_averaging = enable_parameter_averaging
        # 多少步以后才保存模型
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        # 多少步计算log
        self.log_interval = log_interval
        # 误差计算平均值的窗口
        self.loss_averaging_window = loss_averaging_window
        # valid 的batch 是batch_size的倍数
        self.num_validation_batches = num_validation_batches
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.prediction_dir = prediction_dir
        # 如果采用平均参数，模型检查点路径
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'
        # 配置log
        self.init_logging(self.log_dir)
        # 使用pp.pformat 格式化当前对象
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))

        self.graph = self.build_graph()
        self.session = tf.Session(graph=self.graph)
        print('built graph')

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def fit(self):
        with self.session.as_default():
            # 热启动从warm_start_init_step步恢复检查点
            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0
            # 128
            train_generator = self.reader.train_batch_generator(self.batch_size)
            # 验证数据batch_size是训练batch的num_validation_batches倍
            val_generator = self.reader.val_batch_generator(self.num_validation_batches * self.batch_size)
            # 构建连个双端队列 长度100
            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)
            val_score_history = deque(maxlen=self.loss_averaging_window)
            # inf ,0
            best_validation_loss, best_validation_tstep = float('inf'), 0
            restarts = 0
            # 20万个batch
            while step < self.num_training_steps:
                # 验证数
                print('******************{0}'.format(step))
                val_batch_df = next(val_generator)

                val_feed_dict = {
                    # 对每一个数据列，根据列明找到对应的place_holder feed值
                    # 'data', 'is_nan','page_id','access','agent','test_data','test_is_nan'
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in val_batch_df if hasattr(self, placeholder_name)
                }
                # 更新学习率
                val_feed_dict.update({self.learning_rate_var: self.learning_rate})
                # keep_prob 改成1.0 因为训练不用dropout
                if hasattr(self, 'keep_prob'):
                    val_feed_dict.update({self.keep_prob: 1.0})
                # 验证时 不用训练
                if hasattr(self, 'is_training'):
                    val_feed_dict.update({self.is_training: False})
                # debug
                # vlist=[v.shape for v in list(val_feed_dict.values())[:-3]]
                # print(list(zip(val_feed_dict.keys(),vlist)))                # vlist=[v.shape for v in list(val_feed_dict.values())[:-3]]
                # print(list(zip(val_feed_dict.keys(),vlist)))

                [val_loss, score] = self.session.run(
                    fetches=[self.loss, self.score],
                    feed_dict=val_feed_dict
                )
                val_loss_history.append(val_loss)
                val_score_history.append(score)
                # 监控tensor
                if hasattr(self, 'monitor_tensors'):
                    for name, tensor in self.monitor_tensors.items():
                        [np_val] = self.session.run([tensor], feed_dict=val_feed_dict)
                        print(name)
                        print('min', np_val.min())
                        print('max', np_val.max())
                        print('mean', np_val.mean())
                        print('std', np_val.std())
                        print('nans', np.isnan(np_val).sum())

                # 训练步骤
                train_batch_df = next(train_generator)
                train_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)
                }
                train_feed_dict.update({self.learning_rate_var: self.learning_rate})
                if hasattr(self, 'keep_prob'):
                    train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                if hasattr(self, 'is_training'):
                    train_feed_dict.update({self.is_training: True})
                # step 即训练
                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict=train_feed_dict
                )
                train_loss_history.append(train_loss)
                # 10次检查一次最佳 所以用队列存放每次的train_loss
                if step % self.log_interval == 0:
                    # 100次平均
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    avg_val_score = sum(val_score_history) / len(val_score_history)
                    metric_log = (
                        "[[step {:>8}]     "
                        "[train     loss: {:<12}]     "
                        "[val     loss: {:<12} ]    "
                        "[score     score: {:<12}]     "
                    ).format(step, round(avg_train_loss, 8), round(avg_val_loss, 8), round(avg_val_score, 8))
                    logging.info(metric_log)
                    # 最佳平均
                    if avg_val_loss < best_validation_loss:
                        best_validation_loss = avg_val_loss
                        best_validation_tstep = step
                        # 保存最佳模型 至少step >500次
                        if step > self.min_steps_to_checkpoint:
                            self.save(step)
                            if self.enable_parameter_averaging:
                                self.save(step, averaged=True)
                    # 5000 次没有更新最佳平均了，则早停
                    if step - best_validation_tstep > self.early_stopping_steps:
                        #
                        if self.num_restarts is None or restarts >= self.num_restarts:
                            logging.info('best validation loss of {} at training step {}'.format(
                                best_validation_loss, best_validation_tstep))
                            logging.info('early stopping - ending training.')
                            return
                        # 重启2次；每次学习率降低一倍
                        if restarts < self.num_restarts:
                            self.restore(best_validation_tstep)
                            logging.info('halving learning rate')
                            self.learning_rate /= 2.0
                            step = best_validation_tstep
                            restarts += 1

                step += 1
            # <500次 也就是从来没有保存过模型(数据量不足)
            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            logging.info('num_training_steps reached - ending training')

    def predict(self, chunk_size=512):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)

        if hasattr(self, 'prediction_tensors'):
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            test_generator = self.reader.test_batch_generator(chunk_size)
            for i, test_batch_df in enumerate(test_generator):
                test_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in test_batch_df if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    test_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    test_feed_dict.update({self.is_training: False})

                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                np_tensors = self.session.run(
                    fetches=tf_tensors,
                    feed_dict=test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    prediction_dict[tensor_name].append(tensor)
            # 返回预测结果
            result_df = pd.DataFrame()
            for tensor_name, tensor in prediction_dict.items():
                np_tensor = np.concatenate(tensor, 0)
                result_df[tensor_name] = np_tensor
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                np_tensor = tensor.eval(self.session)

                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)
        return result_df

    def save(self, step, averaged=False):
        """

        :param step:
        :param averaged: 是否带滑动平均的梯度 ，保存方法不同
        :return:
        """
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)
        # model_path :./checkpoints/model
        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}'.format(model_path))
        saver.save(self.session, model_path, global_step=step)

    def restore(self, step=None, averaged=False):
        """

        :param step:
        :param averaged:
        :return:
        """
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        # 恢复最新的检查点
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            logging.info('restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)

    def init_logging(self, log_dir):
        """
        配置log
        :param log_dir:
        :return:
        """
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)
        # 重新定义log 配置
        reload(logging)  # bad
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def update_parameters(self, loss):
        """
        step相当于训练
        :param loss:
        :return:
        """
        if self.regularization_constant != 0:
            # 所有训练变量 平方和求根 的平方和-->这个正则项，迫使参数 平方和的根变小
            l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            loss = loss + self.regularization_constant * l2_norm

        optimizer = self.get_optimizer(self.learning_rate_var)
        # list(zip(grads, var_list)) 梯度和变量
        grads = optimizer.compute_gradients(loss)
        # <-20 >20的将会被裁剪（用-20,20替代）
        clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads]
        # 应用梯度下降
        step = optimizer.apply_gradients(clipped, global_step=self.global_step)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                # 执行一组操作，在执行滑动平均前，执行梯度计算
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        logging.info('all parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')  # 所有参数的个数 prod 求乘积
        logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def build_graph(self):
        """
        创建个graph 在此graph上创建整个图，并且返回这个图
        :return:
        """
        with tf.Graph().as_default() as graph:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
            self.global_step = tf.Variable(0, trainable=False)
            # 学习率初始值
            self.learning_rate_var = tf.Variable(0.0, trainable=False)
            # 损失函数
            self.loss = self.calculate_loss()
            self.update_parameters(self.loss)
            # 两个saver 一个有滑动平均 一个没有滑动平均
            self.saver = tf.train.Saver(max_to_keep=1)
            # 初始化saver
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()

            return graph
