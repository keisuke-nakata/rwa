import os

import numpy as np
from keras.models import load_model, Sequential
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rwa import RWA
import datagen

# save_root_dir = '/tmp'
save_root_dir = '/data/keisuke.nakata/rwa'


def seq_length(timestr, model_weight_filename):
    trained_model_path = os.path.join(save_root_dir, 'seq_length', timestr, model_weight_filename)
    trained_model = load_model(trained_model_path, custom_objects={'RWA': RWA})
    trained_model.summary()
    hidden_states = Sequential(trained_model.layers[:2])
    hidden_states.summary()

    figdir = os.path.join(save_root_dir, 'seq_length', timestr, 'figs')
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    randgen = np.random.RandomState(777)
    for length in (1, 10, 50, 100, 200, 300, 400, 450, 499, 500, 501, 550, 600, 700, 800, 900, 1000):
        data = randgen.normal(size=(1, length, 1))
        label = np.asarray([(length > 500)])

        pred = trained_model.predict_on_batch(data)
        states = hidden_states.predict_on_batch(data)

        print('length: {}, pred: {}, gt: {}'.format(length, pred.ravel(), label.ravel()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(states[0])
        ax.set_xlabel('state dim')
        ax.set_ylabel('time dim')
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(os.path.join(figdir, '{}.png'.format(length)))
        plt.close(fig)


def addition(length, timestr, model_weight_filename):
    dirname = 'addition_length{}'.format(length)
    trained_model_path = os.path.join(save_root_dir, dirname, timestr, model_weight_filename)
    trained_model = load_model(trained_model_path, custom_objects={'RWA': RWA})
    trained_model.summary()
    hidden_states = Sequential(trained_model.layers[:2])
    hidden_states.summary()

    figdir = os.path.join(save_root_dir, dirname, timestr, 'figs')
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    test_addition_datagen = datagen.generate_adding_data(batch_size=100, length=length, random_generator=777)
    data, label = next(test_addition_datagen)
    pred = trained_model.predict_on_batch(data)
    states = hidden_states.predict_on_batch(data)

    for i, (data0, label0, pred0, states0) in enumerate(zip(data, label, pred, states)):
        print('pred: {}, gt: {}'.format(pred0.ravel(), label0.ravel()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(states0)
        ax.set_xlabel('state dim')
        ax.set_ylabel('time dim')
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.suptitle('pred: {}, gt: {}, where: {}'.format(pred0.ravel(), label0.ravel(), np.where(data0[:, 0])[0]))
        fig.savefig(os.path.join(figdir, '{}.png'.format(i)))
        plt.close(fig)


if __name__ == '__main__':
    timestr = '20170516_084629'
    model_weight_filename = 'weights.92-0.0168-1.0000.hdf5'
    seq_length(timestr, model_weight_filename)

    # length = 100
    # timestr = '20170516_065502'
    # model_weight_filename = 'weights.91-0.0000.hdf5'
    # addition(length, timestr, model_weight_filename)

    # length = 1000
    # timestr = '20170516_070638'
    # model_weight_filename = 'weights.88-0.0002.hdf5'
    # addition(length, timestr, model_weight_filename)
