from architecture import generator, discriminator
# import tensorflow as tf
import utils
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data_path = "data/Train"
n_epochs =  196#21#98#196
batch_size = 1

# placeholders
a_real = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])  # unfixed-BF image (actual)
b_real = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])  # fixed-BF image (actual)

phase = tf.placeholder(tf.bool, name="phase")  # phase for training

fake_b = generator.generator(a_real, phase_in=phase, scope='a2b')  # generate fake-fixed-BF using real-unfixed BF
fake_a = generator.generator(b_real, phase_in=phase, scope='b2a')  # generate fake-unfixed-BF using real-fixed BF

fake_b_dis = discriminator.discriminator(fake_b, training=phase, scope='b')
fake_a_dis = discriminator.discriminator(fake_a, training=phase, scope='a')

rec_a = generator.generator(fake_b, phase_in=phase, scope='b2a')  # reconstructing unfixed-BF from fake-fixed-BF

rec_b = generator.generator(fake_a, phase_in=phase, scope='a2b')  # reconstructing fixed-BF from fake-unfixed-BF

gen_a2b_loss = tf.reduce_mean(tf.losses.mean_squared_error(fake_b_dis, tf.ones_like(fake_b_dis)))
gen_b2a_loss = tf.reduce_mean(tf.losses.mean_squared_error(fake_a_dis, tf.ones_like(fake_a_dis)))

# cycle_loss_unfixed = tf.reduce_mean(tf.abs(a_real - rec_a))
# cycle_loss_fixed = tf.reduce_mean(tf.abs(b_real - rec_b))

cycle_loss_unfixed = tf.reduce_mean(tf.losses.mean_squared_error(a_real,rec_a))
cycle_loss_fixed = tf.reduce_mean(tf.losses.mean_squared_error(b_real,rec_b))

# final generator loss
g_loss =  (gen_a2b_loss + gen_b2a_loss) + 10 * (cycle_loss_unfixed + cycle_loss_fixed)   # FOR deepDeconv
# g_loss =  (gen_a2b_loss + gen_b2a_loss) + 10 * (cycle_loss_unfixed + cycle_loss_fixed)   # FOR das_despeckle
# g_loss =  (gen_a2b_loss + gen_b2a_loss) + 20 * (cycle_loss_unfixed + cycle_loss_fixed)   # FOR deconv_despeckle
learning_rate=0.0002
# g_loss =  (gen_a2b_loss + gen_b2a_loss) + 20 * (cycle_loss_unfixed + cycle_loss_fixed)   # FOR deepDeconv
# learning_rate=0.0005
# learning_rate=0.0001
# learning_rate=0.00001

a_fake_sample = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
b_fake_sample = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

dis_a_real = discriminator.discriminator(a_real, training=phase, scope='a')
dis_b_real = discriminator.discriminator(b_real, training=phase, scope='b')

dis_a_fake_sample = discriminator.discriminator(a_fake_sample, training=phase, scope='a')
dis_b_fake_sample = discriminator.discriminator(b_fake_sample, training=phase, scope='b')

# # discriminator loss for liver
da_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(dis_a_real, tf.ones_like(dis_a_real)))
da_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(dis_a_fake_sample, tf.zeros_like(dis_a_fake_sample)))
da_loss = (da_loss_real + da_loss_fake) / 2

db_loss_real = tf.reduce_mean(tf.losses.mean_squared_error(dis_b_real, tf.ones_like(dis_b_real)))
db_loss_fake = tf.reduce_mean(tf.losses.mean_squared_error(dis_b_fake_sample, tf.zeros_like(dis_b_fake_sample)))
db_loss = (db_loss_real + db_loss_fake) / 2

# final discriminator loss
d_loss = da_loss + db_loss

# summaries
gen_a2b_loss_sum = tf.summary.scalar("gen_a2b_loss", gen_a2b_loss)
gen_b2a_loss_sum = tf.summary.scalar("gen_b2a_loss", gen_b2a_loss)


cycle_loss_unfixed_sum = tf.summary.scalar("cycle_loss_unfixed", cycle_loss_unfixed)
cycle_loss_fixed_sum = tf.summary.scalar("cycle_loss_fixed", cycle_loss_fixed)

g_loss_sum = tf.summary.scalar("g_loss", g_loss)
g_sum = tf.summary.merge([gen_b2a_loss_sum, g_loss_sum, gen_a2b_loss_sum,
                          cycle_loss_unfixed_sum, cycle_loss_fixed_sum])

d_loss_sum = tf.summary.scalar("d_loss", d_loss)

db_loss_real_sum = tf.summary.scalar("db_loss_real", db_loss_real)
db_loss_fake_sum = tf.summary.scalar("db_loss_fake", db_loss_fake)
db_loss_sum = tf.summary.scalar("db_loss", db_loss)

da_loss_real_sum = tf.summary.scalar("da_loss_real", da_loss_real)
da_loss_fake_sum = tf.summary.scalar("da_loss_fake", da_loss_fake)
da_loss_sum = tf.summary.scalar("da_loss", da_loss)

d_sum = tf.summary.merge([da_loss_sum, da_loss_real_sum, da_loss_fake_sum,
                          d_loss_sum, db_loss_sum, db_loss_real_sum, db_loss_fake_sum])

# optimizer
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
for var in t_vars:
    print(var.name)
# d_optim = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
# g_optim = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)

# read images
train_img, train_labels = utils.get_img_path(data_path)
num_train = int(len(train_img))

# initialize all valuables
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

resume_training = True
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    sess.graph.finalize()
    dir_save = "./training_weights/"
    utils.mkdir(dir_save)

    print(' [*] Loading checkpoint...')
    checkpoint = tf.train.latest_checkpoint(dir_save)
    indicie = int(0)
    if resume_training is False:
        print(' [*] Starting training')
    elif checkpoint:
        print(" [*] Loading succeeds! Copy variables from % s", checkpoint)
        c = checkpoint.find('progress-')
        indicie = int(checkpoint[c + 9:]) + 1
        e_count = int(indicie / num_train)
        print(indicie)
        saver.restore(sess, checkpoint)
    else:
        print(' [*] No checkpoint. Starting training')
        e_count = 0
    print("starting indicie : ", indicie)
    dir_ = "./summaries/"
    utils.mkdir(dir_)
    summary_writer = tf.summary.FileWriter(dir_, sess.graph)
    counter = 0

    for epoch in range(n_epochs):
        ec = e_count + epoch
        for run in range(0, num_train, batch_size):
            a_real_ipt, b_real_ipt = utils.train_next_batch(batch_size, num_train, train_img, train_labels)
            # run g_a2b
            fake_b_, rec_b_ = sess.run([fake_b,rec_b], feed_dict={a_real: a_real_ipt,b_real: b_real_ipt, phase: True})
            # run g_b2a
            fake_a_, rec_a_ = sess.run([fake_a, rec_a], feed_dict={a_real: a_real_ipt,b_real: b_real_ipt, phase: True})

            counter += 1
            _, g_summary_str = sess.run([g_optim, g_sum],
                                        feed_dict={a_real: a_real_ipt, b_real: b_real_ipt, phase: True})

            summary_writer.add_summary(g_summary_str, global_step=indicie + counter)

            _, d_summary_str = sess.run([d_optim, d_sum],
                                        feed_dict={a_real: a_real_ipt, b_real: b_real_ipt,
                                                   a_fake_sample: fake_a_, b_fake_sample: fake_b_, phase: True})

            summary_writer.add_summary(d_summary_str, global_step=indicie + counter)

            print("Epoch: (%2d) (%2d) [%2d / %2d]" % (ec, indicie + counter, run, num_train))
            # sample
            if (indicie + counter) % 127 == 0:#508 == 0:
                save_dir = './progress/'
                utils.mkdir(save_dir)
                data_in = 255 * a_real_ipt
                img_in = data_in.astype(np.uint8)
                data_tar = 255 * b_real_ipt
                img_tar = data_tar.astype(np.uint8)
                data_out = 255 * fake_b_
                img_out = data_out.astype(np.uint8)
                utils.im_write(img_in, '%s/input.bmp' % save_dir)
                utils.im_write(img_tar, '%s/target.bmp' % save_dir)
                utils.im_write(img_out, '%s/output.bmp' % save_dir)
                # save
            if (indicie + counter) % 8128 == 0:
                save_path = saver.save(sess, dir_save + "progress", global_step=indicie + counter)
                print('Model saved in file: % s' % save_path)
                print("Here you can include whatever code you like .....")
