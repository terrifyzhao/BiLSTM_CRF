from bilstm_crf import Model
import tensorflow as tf
import load_data

config = {}
config["lr"] = 0.01
config["embedding_dim"] = 100
config["sen_len"] = 15
config["batch_size"] = 32
config["embedding_size"] = 1856
config["tag_size"] = 27

X, y, seq_len = load_data.train_data()
X_eval, y_eval, seq_len_eval = load_data.eval_data()

model = Model(config)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy],
                                feed_dict={model.keep_prob: 0.9,
                                           model.input_data: X,
                                           model.labels: y,
                                           model.seq_len: seq_len})
        print('epoch: ', epoch, ' loss: ', loss, ' acc: ', acc)

        _, eval_loss, eval_acc = sess.run([model.train_op, model.loss, model.accuracy],
                                          feed_dict={model.keep_prob: 1,
                                                     model.input_data: X_eval,
                                                     model.labels: y_eval,
                                                     model.seq_len: seq_len_eval})
        print('epoch: ', epoch, ' eval_loss: ', eval_loss, ' eval_acc: ', eval_acc)

    saver = tf.train.Saver()
    saver.save(sess, 'BiLSTM_CRF.model', epoch)
