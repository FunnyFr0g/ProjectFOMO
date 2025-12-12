import torch.nn as nn
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
import matplotlib.pyplot as plt
from torchvision.models import resnet18, mobilenet_v2

from FOMOmodels import *
from FOMO_predict_clearml import prepare_image


def calculate_receptive_field(model, input_shape=(3, 224, 224)):
    """
    Ручной расчет глубины воспринимающего поля
    """
    rf = 1  # начальный размер рецептивного поля
    stride_product = 1

    # Пример для последовательных слоев
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            layers.append((name, module))

    for name, layer in layers:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation

            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            if isinstance(stride, tuple):
                stride = stride[0]
            if isinstance(padding, tuple):
                padding = padding[0]
            if isinstance(dilation, tuple):
                dilation = dilation[0]

            rf = rf + (kernel_size - 1) * stride_product
            stride_product *= stride

        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation if hasattr(layer, 'dilation') else 1

            if isinstance(kernel_size, (int, tuple)):
                kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            if isinstance(stride, (int, tuple)):
                stride = stride if isinstance(stride, int) else stride[0]
            if isinstance(padding, (int, tuple)):
                padding = padding if isinstance(padding, int) else padding[0]

            rf = rf + (kernel_size - 1) * stride_product
            stride_product *= stride

    return rf, stride_product


def visualize_receptive_field(model, input_size=(3, 224, 224), img_path=None):
    """
    Визуализация воспринимающего поля через анализ градиентов
    """
    # Создаем тестовый тензор
    if img_path is None:
        x = torch.randn(1, *input_size, requires_grad=True)
    else:
        x, _ = prepare_image(img_path, input_size[1:])
        x = x.requires_grad_(requires_grad=True)

    print(x.shape)

    # Проход вперед
    output = model(x)

    # Выбираем один выходной пиксель
    target_output = output[0, 0, output.shape[2] // 2, output.shape[3] // 2]
    # target_output = output[0, 0, 55//4, 65//4]

    # Обратное распространение
    target_output.backward()

    # Градиенты покажут, какие входные пиксели влияют на выход
    gradients = x.grad[0].abs().sum(dim=0)
    print(f'{x.grad.shape = }')

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gradients.detach().numpy(), cmap='hot')
    plt.title('Градиенты по входу')
    plt.colorbar()

    # Бинаризация для определения границ RF
    plt.subplot(1, 2, 2)
    threshold = gradients.max() * 0.1
    mask = (gradients > threshold).float()
    plt.imshow(mask.detach().numpy(), cmap='gray')
    plt.title('Область влияния (рецептивное поле)')

    return gradients, mask


if __name__ == '__main__':
    input_shape = (3, 4*224, 4*224)

    checkpoint_path = r'weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth'

    model = FomoModel56()
    # model = FomoModel112()
    # model = FomoModelResV0()
    # model = FomoModelResV1()

    # state_dict = torch.load(checkpoint_path)
    # model.load_state_dict(state_dict)
    # model = resnet18()
    result = calculate_receptive_field(model, input_shape)
    print(result)
    img_path = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_54a2a3bf4b814f6780c5e1045f2b24ec\val\images\20190925_101846_1_1_visible_frame_0769_drone.png'
    gradients, mask = visualize_receptive_field(model, input_shape, img_path)

    plt.show()
