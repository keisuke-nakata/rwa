import numpy as np


def _get_random_generator(r):
    if r is None:
        randgen = np.random.RandomState()
    elif isinstance(r, np.random.RandomState):
        randgen = r
    elif isinstance(r, int):
        randgen = np.random.RandomState(r)
    else:
        raise ValueError('type {} is not allowed'.format(type(r)))
    return randgen


def generate_sequence_length_data(batch_size=1, max_length=1000, random_generator=None):
    if batch_size != 1:
        raise ValueError('batch_size must be 1 because of implementation limitation')
    randgen = _get_random_generator(random_generator)
    while True:
        t = randgen.randint(low=1, high=max_length + 1)
        data = randgen.normal(size=(1, t, 1))
        label = np.asarray([(t > max_length / 2)])
        yield (data, label)


def generate_adding_data(batch_size, length, random_generator=None):
    randgen = _get_random_generator(random_generator)
    while True:
        data = np.empty((batch_size, length, 2))
        label = np.empty(batch_size)
        for batch in range(batch_size):
            where = randgen.choice(length, size=2, replace=False)
            flags = np.zeros(length)
            flags[where] = 1.0

            seq = randgen.uniform(low=0.0, high=1.0, size=length)

            addition = flags @ seq

            data[batch, :, 0] = flags
            data[batch, :, 1] = seq
            label[batch] = addition
        yield (data, label)
