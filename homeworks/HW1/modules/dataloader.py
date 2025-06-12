import numpy as np


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__
        self._indices = np.arange(len(X))

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return (self.num_samples() + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return len(self.X)

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        if self.shuffle:
            np.random.shuffle(self._indices)
        self.batch_id = 0
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id >= len(self):
            raise StopIteration
            
        start_idx = self.batch_id * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples())
        batch_indices = self._indices[start_idx:end_idx]
        
        x_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        self.batch_id += 1
        return x_batch, y_batch
