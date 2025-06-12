import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = np.dot(input, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        # bcs: dl/dx = dl/df * df/dx = dl/df * W
        return np.dot(grad_output, self.weight)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        # bcs: dl/dw = (dl/df)^T * x
        self.grad_weight += np.dot(grad_output.T, input)
        if self.bias is not None:
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={self.bias is not None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # 1. Вычисляем среднее по батчу
            self.mean = np.mean(input, axis=0)  # shape: (num_features,)
            
            # 2. Центрируем данные
            self.input_mean = input - self.mean  # shape: (batch_size, num_features)
            
            # 3. Вычисляем дисперсию
            self.var = np.mean(self.input_mean ** 2, axis=0)  # shape: (num_features,)
            
            # 4. Нормируем вход с учетом статистик
            self.sqrt_var = np.sqrt(self.var + self.eps)  # shape: (num_features,)
            self.inv_sqrt_var = 1.0 / self.sqrt_var  # shape: (num_features,)
            self.norm_input = self.input_mean * self.inv_sqrt_var  # shape: (batch_size, num_features)
            
            # 5. Обновляем бегущие статистики
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var * (input.shape[0] / (input.shape[0] - 1))
        else:
            # В режиме eval используем бегущие статистики
            self.input_mean = input - self.running_mean
            self.inv_sqrt_var = 1.0 / np.sqrt(self.running_var + self.eps)
            self.norm_input = self.input_mean * self.inv_sqrt_var
        
        # 6. Применяем аффинное преобразование если нужно
        output = self.norm_input
        if self.affine:
            output = output * self.weight + self.bias
        
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        # Если есть аффинное преобразование, умножаем градиент на веса
        if self.affine:
            grad_output = grad_output * self.weight
        
        if self.training:
            grad_norm = grad_output * self.inv_sqrt_var
            grad_mean = np.mean(grad_norm, axis=0)
            grad_std = np.mean(grad_norm * self.norm_input, axis=0)
            
            # Финальная формула
            grad_input = grad_norm - grad_mean - self.norm_input * grad_std
        else:
            grad_input = grad_output * self.inv_sqrt_var
        
        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            # Генерируем маску с вероятностью (1-p)
            self.mask = (np.random.random(input.shape) > self.p).astype(float)
            # Нормируем на (1-p) чтобы сохранить среднее значение
            return input * self.mask / (1 - self.p)
        else:
            # В режиме eval просто пропускаем вход
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            # В режиме train умножаем градиент на маску и нормируем
            return grad_output * self.mask / (1 - self.p)
        else:
            # В режиме eval просто пропускаем градиент
            return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        # Список для хранения промежуточных входов
        self.forward_cache = []  

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        # Очищаем кэш перед новым проходом
        self.forward_cache = []
        
        # Применяем все модули последовательно
        current_output = input  # вход модели
        for module in self.modules:
            self.forward_cache.append(current_output)
            current_output = module(current_output)
            
        return current_output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        # Проходим по модулям в обратном порядке
        current_grad = grad_output
        for module, module_input in zip(reversed(self.modules), reversed(self.forward_cache)):
            # Обновляем градиенты параметров
            module.update_grad_parameters(module_input, current_grad)
            # Вычисляем градиент по входу
            current_grad = module.compute_grad_input(module_input, current_grad)
            
        return current_grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
