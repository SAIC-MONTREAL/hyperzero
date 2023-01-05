import torch


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Code based on: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5
    """
    def __init__(self, *tensors, device='cpu', batch_size=256, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        Each tensor is in the form of (N, D) with N the number
        of datapoints and D the dimension of the data.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.device = device

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class FastTensorMetaDataLoader:
    """
    Fast tensor dataloader for meta learning.
    """
    def __init__(self, *tensors, device='cpu', batch_size=256, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        Each tensor is in the form of (T, N, D) with T the number of tasks,
        N the number of datapoints and D the dimension of the data.

        :param *tensors: tensors to store. Must have the same length @ dim 0 and @ dim 1.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        assert all(t.shape[1] == tensors[0].shape[1] for t in tensors)
        self.tensors = tensors
        self.device = device

        self.dataset_len = self.tensors[0].shape[1]
        self._n_tasks = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def shuffle_indices(self):
        self._shuffle_indices()
        self.i = 0

    def _shuffle_indices(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = None

    def sample(self, task_id):
        assert task_id < self._n_tasks

        if self.i >= self.dataset_len:
            self._shuffle_indices()
            self.i = 0

        task_tensors = tuple(t[task_id] for t in self.tensors)

        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in task_tensors)
        else:
            batch = tuple(t[self.i:self.i + self.batch_size] for t in task_tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def n_tasks(self):
        return self._n_tasks
