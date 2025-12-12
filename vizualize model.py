import torch
import torch.nn as nn
from torchviz import make_dot
from graphviz import Digraph
from collections import OrderedDict
import numpy as np

from FOMOmodels import *

def visualize_model_ascii(model, input_size=(1, 3, 224, 224)):
    """
    Простая ASCII визуализация структуры модели
    """
    print("=" * 80)
    print("АРХИТЕКТУРА МОДЕЛИ")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Общее число параметров: {total_params:,}")
    print(f"Обучаемые параметры: {trainable_params:,}")
    print(f"Необучаемые параметры: {total_params - trainable_params:,}")
    print("-" * 80)

    # Собираем информацию о слоях
    layers_info = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(layers_info)

            # Получаем размеры
            if isinstance(output, (list, tuple)):
                output_shape = [tuple(o.shape) if hasattr(o, 'shape') else str(o)
                                for o in output]
            elif hasattr(output, 'shape'):
                output_shape = tuple(output.shape)
            else:
                output_shape = str(output)

            layers_info.append(OrderedDict([
                ("layer_idx", module_idx),
                ("type", class_name),
                ("input_shape", input[0].shape if isinstance(input, tuple) else input.shape),
                ("output_shape", output_shape),
                ("params", sum(p.numel() for p in module.parameters())),
                ("trainable", any(p.requires_grad for p in module.parameters()))
            ]))

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            return hook

    # Регистрируем хуки
    hooks = []
    for name, module in model.named_modules():
        hook = register_hook(module)
        if hook:
            hooks.append(module.register_forward_hook(hook))

    # Запускаем forward pass
    with torch.no_grad():
        x = torch.randn(*input_size)
        _ = model(x)

    # Удаляем хуки
    for h in hooks:
        h.remove()

    # Выводим таблицу
    print(f"{'ID':<4} {'Тип слоя':<25} {'Вход':<25} {'Выход':<25} {'Параметры':<15} {'Обучаемый'}")
    print("-" * 120)

    for layer in layers_info:
        print(f"{layer['layer_idx']:<4} "
              f"{layer['type']:<25} "
              f"{str(layer['input_shape']):<25} "
              f"{str(layer['output_shape']):<25} "
              f"{layer['params']:<15,} "
              f"{'✓' if layer['trainable'] else '✗'}")

    print("=" * 80)

    return layers_info

if __name__ == '__main__':
    model = FomoModel56()
    visualize_model_ascii(model)
