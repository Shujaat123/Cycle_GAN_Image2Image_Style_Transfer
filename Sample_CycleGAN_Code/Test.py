import tensorflow as tf
import utils
import PIL.Image as Image  # , ImageEnhance
# from architecture import generator, discriminator
from architecture import generator, discriminator
from scipy.io import savemat
import os
import numpy as np
import scipy.io
# placeholders
a_real = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # unfixed-BF image (actual)
b_real = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # fixed-BF image (actual)

phase = tf.placeholder(tf.bool, name="phase")  # phase for training

fake_b = generator.generator(a_real, phase_in=phase, scope='a2b')  # generate fake-fixed-BF using real-unfixed BF


# optimizer
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
for var in t_vars:
    print(var.name)

# initialize all valuables
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    # exp_name = 'deepDeconv_PatchGAN_LS_Unet3_8x'
    # exp_name = 'das_despeckle_PatchGAN_LS_Unet3_8x'
    # exp_name = 'deconv_despeckle_PatchGAN_LS_Unet3_8x'
    # exp_name = 'deconv_despeckle_mixed_PatchGAN_LS_Unet3_8x'
    exp_name = 'deepDeconv_PatchGAN_LS_Unet3_8x_PSF2'

    # exp_name = 'PW_deepDeconv_PatchGAN_LS_Unet3_8x'
    # exp_name = 'PW_das_despeckle_PatchGAN_LS_Unet3_8x'
    # exp_name = 'PW_deconv_despeckle_PatchGAN_LS_Unet3_8x'
    # exp_name = 'PW_deconv_despeckle_mixed_PatchGAN_LS_Unet3_8x'

    # exp_name = 'cardiac_MLA_PatchGAN_LS_Unet3_8x'
    # exp_name = 'cardiac_MLA_PatchGAN_LS_Unet3_16x'
    # exp_name = 'cardiac_MLA_PatchGAN_LS_Unet3_64x'

    # exp_name = 'deepDeconv_Unet3_8x'
    # exp_name = 'das_despeckle_Unet3_8x'
    # exp_name = 'deconv_despeckle_Unet3_8x'
    # exp_name = 'deconv_despeckle_mixed_Unet3_8x'
    # exp_name = 'PW_despeckle_Unet3_8x'
    # exp_name = 'PW_deepDeconv_Unet3_8x'
    # exp_name = 'PW_deconv_despeckle_Unet3_8x'
    # exp_name = 'PW_deconv_despeckle_mixed_Unet3_8x'


    dir_save = './training_weights/' + exp_name + '/'
    print(' [*] Loading checkpoint...')
    # checkpoint = tf.train.latest_checkpoint(dir_save)
    # checkpoint = dir_save + 'progress-8128'
    # checkpoint = dir_save + 'progress-16256'
    checkpoint = dir_save + 'progress-20320'
    # checkpoint = dir_save + 'progress-24384'
    # checkpoint = dir_save + 'progress-32512'
    # checkpoint = dir_save + 'progress-40640'
    # checkpoint = dir_save + 'progress-48768'
    # checkpoint = dir_save + 'progress-56896'
    # checkpoint = dir_save + 'progress-65024'
    # checkpoint = dir_save + 'progress-73152'   #  -- best Deconv 21.23494838	0.763655454 -- best das_despeckle 27.37953858	0.91882163
    # checkpoint = dir_save + 'progress-81280'
    # checkpoint = dir_save + 'progress-97536' # --best deconv_despeckle_mixed -- best deconv_despeckle 22.3554383	0.78069997 ///// 17.7784    0.7408   21.1974    0.8066 deepDeconv
    # checkpoint = dir_save + 'progress-105664'
    # checkpoint = dir_save + 'progress-138176' # 17.9805    0.6908   21.3911    0.7264 despeckle_deconv
    # checkpoint = dir_save + 'progress-146304' # 17.9805    0.6908   21.3911    0.7264 despeckle_deconv
    # checkpoint = dir_save + 'progress-186944'  # 26.8353    0.7307   26.7784    0.8989 das_despeckle
    # checkpoint = dir_save + 'progress-195072' # 26.8353    0.7307   26.7784    0.8989 das_despeckle
    # checkpoint = dir_save + 'progress-203200' # 26.8353    0.7307   26.7784    0.8989 das_despeckle
    # checkpoint = dir_save + 'progress-211328'  # 26.8353    0.7307   26.7784    0.8989 das_despeckle
    # checkpoint = dir_save + 'progress-243840' # 26.8353    0.7307   26.4471    0.8996
    # checkpoint = dir_save + 'progress-292608' # 26.8353    0.7307   26.3027    0.8994
    # checkpoint = dir_save + 'progress-341376'
    # checkpoint = dir_save + 'progress-203200'
    # checkpoint = dir_save + 'progress-390144'
    # checkpoint = dir_save + 'progress-585216'


    print(checkpoint)
    saver.restore(sess, checkpoint)
   
    input_path   = './data/Test/input_das/'
    # input_path = './data/Test/input_das_PWsALL/'
    # input_path = './data/Test/input_cardiac_MLA/'
    # input_path = './data/Test/input_phantom_resolution/'
    label_path  = './data/Test/label_deepDeconv/'
    # label_path  = './data/Test/label_das_despeckle/'
    # label_path  = './data/Test/label_deconv_despeckle/'
    # label_path = './data/Test/input_das_PWsALL/'
    # label_path = './data/Test/label_cardiac_MLA/'
    # label_path = './data/Test/label_phantom_resolution/'

    sub_list   = os.listdir(input_path)
    sub_num    = np.shape(sub_list)[0]

    counter = 0
    save_dir = './data/Test/output_' + exp_name
    # utils.mkdir(save_dir)
    for sub_loop in range(sub_num):
        print(sub_list[sub_loop])
        sub_name = sub_list[sub_loop]
        sub_path = input_path + sub_name

        BP_list = os.listdir(sub_path)
        BP_num = np.shape(BP_list)[0]

        for bp_loop in range(BP_num):
            print(BP_list[bp_loop])
            BP_name = BP_list[bp_loop]
            bp_path = sub_path + '/' + BP_name

            frame_list = os.listdir(bp_path)
            fr_num = np.shape(frame_list)[0]

            save_dir_ = save_dir + '/' + sub_name + '/' + BP_name
            utils.mkdir(save_dir_)
            data_full_path = []
            label_full_path =[]
            for fr_loop in range(fr_num):
                print(frame_list[fr_loop])
                fr_name = frame_list[fr_loop]
                fr_path = bp_path + '/' + fr_name
                data_full_path = fr_path
                label_full_path = label_path + sub_name + '/' + BP_name + '/' + fr_name
                target_size = (256, 256)

                img_in = Image.open(data_full_path)
                # img_in = img_in.resize(target_size, Image.ANTIALIAS)
                input_images = np.asarray(img_in).astype(np.float32)/255
                # min_array = np.min(input_images)
                # max_array = np.max(input_images)
                # input_images = utils.normalize(input_images, min_array, max_array)
                # print('input-image-(MIN:', np.min(input_images), ',MAX:', np.max(input_images), ')')

                img_lab = Image.open(label_full_path)
                # img_lab = img_lab.resize(target_size, Image.ANTIALIAS)
                target_images = np.asarray(img_lab).astype(np.float32)/255
                # min_array = np.min(target_images)
                # max_array = np.max(target_images)
                # target_images = utils.normalize(target_images, min_array, max_array)
                # print('target-image-(MIN:',np.min(target_images),',MAX:',np.max(target_images),')')

                a_real_ipt = np.asarray(input_images)
                a_real_ipt = np.expand_dims(a_real_ipt, axis=0)
                a_real_ipt = np.expand_dims(a_real_ipt, axis=3)

                b_real_ipt = np.asarray(target_images)
                b_real_ipt = np.expand_dims(b_real_ipt, axis=0)
                b_real_ipt = np.expand_dims(b_real_ipt, axis=3)


                # run g_a2b
                fake_b_ = sess.run(fake_b, feed_dict={a_real: a_real_ipt, phase: True})
                # min_array = np.min(fake_b_)
                # max_array = np.max(fake_b_)
                # fake_b_ = utils.normalize(fake_b_, min_array, max_array)
                # print('output-image-(MIN:', np.min(fake_b_), ',MAX:', np.max(fake_b_), ')')

                data_in = 255 * np.squeeze(a_real_ipt)
                img_in = data_in.astype(np.uint8)
                data_lab = 255 * np.squeeze(b_real_ipt)
                img_lab = data_lab.astype(np.uint8)

                data_out = 255 * np.squeeze(fake_b_)
                img_out = data_out.astype(np.uint8)
                # data_out = np.squeeze(fake_b_)
                # img_out = data_out

                utils.im_write(img_out, save_dir_ + '/' + fr_name)
                # utils.im_write((255 *img_out).astype(np.uint8), save_dir_ + '/' + fr_name)
                filename_ = save_dir_ + '/' + fr_name[:-4] + '.mat'
                savemat(filename_, {'input_img': img_in,'label_img': img_lab,'output_img': img_out})
