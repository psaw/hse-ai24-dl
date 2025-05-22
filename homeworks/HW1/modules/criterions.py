import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # Вычисляем среднеквадратичную ошибку
        return np.mean((input - target) ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # Градиент MSE: 2(x - y)/(B*N)
        return 2 * (input - target) / np.prod(input.shape)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size = input.shape[0]
        # Применяем LogSoftmax к логитам
        log_probs = self.log_softmax(input)
        # Выбираем вероятности правильных классов
        correct_log_probs = log_probs[np.arange(batch_size), target]
        # Возвращаем среднее отрицательное правдоподобие
        return -np.mean(correct_log_probs)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size = input.shape[0]
        # Получаем вероятности классов
        probs = np.exp(self.log_softmax(input))
        # Создаем градиент: -1/N для правильных классов
        grad_input = probs.copy()
        grad_input[np.arange(batch_size), target] -= 1
        return grad_input / batch_size
