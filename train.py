from bilstm_crf import Model
import tensorflow as tf
import load_data

config = {}
config["lr"] = 0.001
config["embedding_dim"] = 100
config["sen_len"] = 15
config["batch_size"] = 32
config["embedding_size"] = 1856
config["tag_size"] = 27

X, y = load_data.load_data()

model = Model(config)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy],
                                feed_dict={model.keep_prob: 0.9,
                                           model.input_data: X,
                                           model.labels: y})

        # print('crl tag: ', tag)
        # print('crl score: ', score)
        print('epoch: ', epoch, ' loss: ', loss)
        print('epoch: ', epoch, ' crl acc: ', acc)
