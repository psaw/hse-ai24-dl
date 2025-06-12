import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # ReLU(x) = max(0, x)
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # Градиент ReLU: 1 если x > 0, иначе 0
        return grad_output * (input > 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # sigmoid(x) = 1 / (1 + exp(-x))
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # Градиент sigmoid: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_output = self.compute_output(input)
        return grad_output * sigmoid_output * (1 - sigmoid_output)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # Для численной стабильности вычитаем максимум
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_input / np.sum(exp_input, axis=1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # Получаем выход softmax
        softmax_output = self.compute_output(input)
        
        # Векторизованная версия градиента softmax
        # Для каждого примера в батче:
        # grad_input = softmax * (grad_output - sum(softmax * grad_output))
        sum_softmax_grad = np.sum(softmax_output * grad_output, axis=1, keepdims=True)
        return softmax_output * (grad_output - sum_softmax_grad)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # Для численной стабильности вычитаем максимум
        input_max = np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(input - input_max)
        log_sum_exp = np.log(np.sum(exp_input, axis=1, keepdims=True))
        return input - input_max - log_sum_exp

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # Получаем выход softmax
        softmax_output = np.exp(self.compute_output(input))
        
        # Градиент logsoftmax: grad_output - softmax * sum(grad_output)
        sum_grad = np.sum(grad_output, axis=1, keepdims=True)
        return grad_output - softmax_output * sum_grad
