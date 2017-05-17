import datetime as dt
import os

from keras.callbacks import CSVLogger, ModelCheckpoint

import modelfactory
import datagen

# save_root_dir = '/tmp'
save_root_dir = '/data/keisuke.nakata/rwa'


def seq_length(rwa):
    units = 250
    batch_size = 1

    model = modelfactory.build_seq_length_model(units=units, batch_size=batch_size, rwa=rwa)
    model.summary()

    train_sequence_length_datagen = datagen.generate_sequence_length_data(batch_size=batch_size, max_length=1000, random_generator=0)
    valid_sequence_length_datagen = datagen.generate_sequence_length_data(batch_size=batch_size, max_length=1000, random_generator=0)

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    dirname = 'seq_length/{}'.format(now)
    if not rwa:
        dirname = 'lstm_' + dirname
    save_dir = os.path.join(save_root_dir, dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'))
    checkpoint_path = os.path.join(save_dir, 'weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    callbacks = [csv_logger, model_checkpoint]

    model.fit_generator(
        generator=train_sequence_length_datagen,
        steps_per_epoch=10 * 100,
        epochs=99,
        callbacks=callbacks,
        validation_data=valid_sequence_length_datagen,
        validation_steps=100)


def addition(length, rwa):
    units = 250
    batch_size = 100

    model = modelfactory.build_addition_model(units=units, batch_size=batch_size, rwa=rwa)
    model.summary()

    train_addition_datagen = datagen.generate_adding_data(batch_size=batch_size, length=length, random_generator=0)
    valid_addition_datagen = datagen.generate_adding_data(batch_size=batch_size, length=length, random_generator=0)

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    dirname = 'addition_length{}/{}'.format(length, now)
    if not rwa:
        dirname = 'lstm_' + dirname
    save_dir = os.path.join(save_root_dir, dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'))
    checkpoint_path = os.path.join(save_dir, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    callbacks = [csv_logger, model_checkpoint]

    model.fit_generator(
        generator=train_addition_datagen,
        steps_per_epoch=100,
        epochs=99,
        callbacks=callbacks,
        validation_data=valid_addition_datagen,
        validation_steps=1)


if __name__ == '__main__':
    # seq_length()
    # addition(length=100)
    # addition(length=1000)

    seq_length(rwa=False)
    # addition(length=100, rwa=False)
    # addition(length=1000, rwa=False)
