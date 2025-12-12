from graphviz import Digraph

from FOMOmodels import *
from FOMO_predict_clearml import prepare_image

def visualize_model_graphviz(model, input_size=(1, 3, 224, 224), filename='model_architecture'):
    """
    Создание графического представления модели
    """
    # Создаем граф
    dot = Digraph(comment='Model Architecture',
                  format='png',
                  graph_attr={'rankdir': 'TB',  # TB - сверху вниз, LR - слева направо
                              'splines': 'ortho',
                              'nodesep': '0.5',
                              'ranksep': '0.8'},
                  node_attr={'style': 'filled',
                             'shape': 'box',
                             'align': 'left',
                             'fontsize': '10',
                             'fontname': 'Helvetica'})

    # Карта цветов для типов слоев
    layer_colors = {
        'Conv2d': '#FFCCCC',  # красный
        'Linear': '#CCFFCC',  # зеленый
        'BatchNorm': '#CCCCFF',  # синий
        'ReLU': '#FFFFCC',  # желтый
        'Pool': '#FFCCFF',  # фиолетовый
        'Dropout': '#CCFFFF',  # голубой
        'default': '#E0E0E0'  # серый
    }

    # Функция для получения цвета слоя
    def get_layer_color(layer_type):
        for key, color in layer_colors.items():
            if key in layer_type:
                return color
        return layer_colors['default']

    # Проходим по всем слоям
    node_counter = 0
    layer_nodes = {}

    # Добавляем входной узел
    dot.node('input', f'Input\n{input_size}',
             shape='ellipse', color='#FFE4B5', style='filled')

    # Рекурсивная функция для обхода модулей
    def add_module(module, name, parent_name=None):
        nonlocal node_counter

        module_type = module.__class__.__name__
        node_id = f'node_{node_counter}'
        node_counter += 1

        # Собираем информацию о модуле
        info_lines = [f'{module_type}', f'name: {name}']

        # Добавляем параметры для определенных типов слоев
        if isinstance(module, nn.Conv2d):
            info_lines.append(f'in: {module.in_channels}, out: {module.out_channels}')
            info_lines.append(f'kernel: {module.kernel_size}')
            info_lines.append(f'stride: {module.stride}, padding: {module.padding}')
            if module.groups > 1:
                info_lines.append(f'groups: {module.groups}')

        elif isinstance(module, nn.Linear):
            info_lines.append(f'in: {module.in_features}, out: {module.out_features}')

        elif isinstance(module, nn.BatchNorm2d):
            info_lines.append(f'features: {module.num_features}')
            info_lines.append(f'eps: {module.eps}')

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            info_lines.append(f'kernel: {module.kernel_size}')
            info_lines.append(f'stride: {module.stride}')

        elif isinstance(module, nn.Dropout):
            info_lines.append(f'p: {module.p}')

        # Количество параметров
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            info_lines.append(f'params: {num_params:,}')

        # Создаем узел
        label = '\n'.join(info_lines)
        color = get_layer_color(module_type)

        dot.node(node_id, label, color=color)
        layer_nodes[name] = node_id

        # Добавляем связь с родителем
        if parent_name and parent_name in layer_nodes:
            dot.edge(layer_nodes[parent_name], node_id)

        # Рекурсивно обходим дочерние модули
        for child_name, child_module in module.named_children():
            if child_name:  # Пропускаем пустые имена
                add_module(child_module, f'{name}.{child_name}', name)

    # Начинаем обход с корня
    for name, module in model.named_children():
        add_module(module, name)

    # Добавляем выходной узел
    dot.node('output', 'Output', shape='ellipse', color='#98FB98', style='filled')

    # Соединяем последний слой с выходом
    if layer_nodes:
        last_node = list(layer_nodes.values())[-1]
        dot.edge(last_node, 'output')

    # Сохраняем и отображаем
    dot.render(filename, view=True, cleanup=True)
    print(f"График сохранен как {filename}.png")

    return dot


if __name__ == '__main__':
    model = FomoModel56()
    visualize_model_graphviz(model)