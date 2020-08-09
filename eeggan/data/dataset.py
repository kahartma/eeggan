#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import Iterable, TypeVar, Generic

from braindecode.datautil.signal_target import SignalAndTarget

T = TypeVar('T')


class Data(SignalAndTarget, Iterable, Generic[T]):
    def __init__(self, X: T, y: T, y_onehot: T):
        super().__init__(X, y)
        self.X: T = X
        self.y: T = y
        self.y_onehot: T = y_onehot

    def __iter__(self) -> (T, T, T):
        for i in range(len(self.X)):
            yield self[i]

    def __getitem__(self, index: int) -> (T, T, T):
        return self.X[index], self.y[index], self.y_onehot[index]

    def __len__(self) -> int:
        return len(self.X)


class Dataset(Generic[T]):
    def __init__(self, train_data: Data[T], test_data: Data[T]):
        self.train_data = train_data
        self.test_data = test_data
