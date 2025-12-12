from graphviz import Digraph

from FOMOmodels import *
from FOMO_predict_clearml import prepare_image

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

from FOMOmodels import *
from FOMO_predict_clearml import prepare_image, base as base_model, res_v0_focal

def visualize_activations(model, input_tensor, layer_names=None):
    """
    Визуализация активаций в разных слоях модели
    """
    # Сохраняем активации
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    # Регистрируем хуки
    hooks = []
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Удаляем хуки
    for hook in hooks:
        hook.remove()

    # Визуализируем активации
    n_layers = len(activations)
    cols = 4
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for idx, (layer_name, activation) in enumerate(activations.items()):
        ax = axes[idx]

        # Разные типы визуализации для разных размерностей
        if activation.dim() == 4:  # Conv активации [batch, channels, H, W]
            # Берем среднее по каналам для первого изображения в батче
            act_mean = activation[0].mean(dim=0).cpu().numpy()
            im = ax.imshow(act_mean, cmap='viridis', aspect='auto')
            ax.set_title(f"{layer_name}\n{activation.shape}")
            plt.colorbar(im, ax=ax)

        elif activation.dim() == 2:  # Linear активации
            act_data = activation[0].cpu().numpy()
            ax.bar(range(len(act_data)), act_data)
            ax.set_title(f"{layer_name}\n{activation.shape}")
            ax.set_xlabel("Нейроны")
            ax.set_ylabel("Активация")

        else:
            # Гистограмма для других типов
            act_flat = activation.flatten().cpu().numpy()
            ax.hist(act_flat, bins=50, alpha=0.7)
            ax.set_title(f"{layer_name}\n{activation.shape}")
            ax.set_xlabel("Значение активации")
            ax.set_ylabel("Частота")

    # Скрываем пустые subplots
    for idx in range(len(activations), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    return activations


if __name__ == '__main__':
    image_path = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_54a2a3bf4b814f6780c5e1045f2b24ec\val\images\20190925_101846_1_1_visible_frame_0769_drone.png'
    image_tensor, _ = prepare_image(image_path)

    mega_model = res_v0_focal

    model = mega_model.model
    state_dict = torch.load(mega_model.checkpoint)
    model.load_state_dict(state_dict)

    visualize_activations(model, image_tensor)