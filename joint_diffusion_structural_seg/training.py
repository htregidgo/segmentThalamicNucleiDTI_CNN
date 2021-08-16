import os
import numpy as np
import keras.callbacks as KC
from keras.optimizers import Adam
from joint_diffusion_structural_seg.generators import image_seg_generator, image_seg_generator_rgb
from joint_diffusion_structural_seg import models
from joint_diffusion_structural_seg import metrics


def train(training_dir,
             path_label_list,
             model_dir,
             batchsize=1,
             crop_size=128,
             scaling_bounds=0.15,
             rotation_bounds=15,
             max_noise_std=0.1,
             max_noise_std_fa=0.03,
             gamma_std=0.1,
             contrast_std=0.1,
             brightness_std=0.1,
             randomize_resolution=False,
             generator_mode='rgb',
             diffusion_resolution=None,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             dropout=0,
             activation='elu',
             lr=1e-4,
             lr_decay=0,
             wl2_epochs=5,
             dice_epochs=200,
             steps_per_epoch=1000,
             checkpoint=None):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)
    # check generator mode
    assert (generator_mode == 'fa_v1') | (generator_mode == 'rgb'), \
        'generator mode must be fa_v1 or rgb'

    if diffusion_resolution is not None:
        if type(diffusion_resolution) == int:
            diffusion_resolution = [diffusion_resolution] * 3

    if generator_mode == 'rgb':
        generator = image_seg_generator_rgb(training_dir,
                                path_label_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution)
    else:
        generator = image_seg_generator(training_dir,
                                path_label_list,
                                batchsize=batchsize,
                                scaling_bounds=scaling_bounds,
                                rotation_bounds=rotation_bounds,
                                max_noise_std=max_noise_std,
                                max_noise_std_fa=max_noise_std_fa,
                                gamma_std=gamma_std,
                                contrast_std=contrast_std,
                                brightness_std=brightness_std,
                                crop_size=crop_size,
                                randomize_resolution=randomize_resolution,
                                diffusion_resolution=diffusion_resolution)

    label_list = np.sort(np.load(path_label_list)).astype(int)
    n_labels = np.size(label_list)

    if crop_size is None:
        aux = next(generator)
        crop_size = aux[0].shape[1:-1]
    if type(crop_size) == int:
        crop_size = [crop_size] * 3

    unet_input_shape = [*crop_size, 5]

    unet_model = models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=None)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        train_model(wl2_model, generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, 'wl2', n_labels, checkpoint)
        checkpoint = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    train_model(unet_model, generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, 'dice', n_labels, checkpoint)

    print('All done!')


def train_model(model,
                generator,
                learning_rate,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                n_labels,
                path_checkpoint=None):

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    if os.path.exists(log_dir) is False:
        os.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % metric_type)
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1)]

    # TensorBoard callback
    if metric_type == 'dice':
        callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    if path_checkpoint is not None:
            print('loading weights from checkpoint:')
            print(path_checkpoint)
            model.load_weights(path_checkpoint, by_name=True)

    # compile
    if metric_type == 'dice':
        model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                  loss=metrics.DiceLoss().loss,
                  loss_weights=[1.0])
    else:
        model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                      loss=metrics.WL2Loss(3.0, n_labels, background_weight=0.01).loss, # since we crop, 0.01 is ok
                      loss_weights=[1.0])

    # fit
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=0)


