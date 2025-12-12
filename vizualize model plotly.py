from graphviz import Digraph

from FOMOmodels import *
from FOMO_predict_clearml import prepare_image

import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict


def visualize_model_interactive(model, input_size=(1, 3, 224, 224)):
    """
    Интерактивная 3D визуализация архитектуры модели
    """
    print("Сбор информации о модели...")

    # Собираем информацию о слоях
    layers_data = []
    module_counter = defaultdict(int)

    def analyze_module(module, name, path="", depth=0):
        module_type = module.__class__.__name__
        module_counter[module_type] += 1

        # Уникальный идентификатор для модуля
        if not name:
            name = f"{module_type}_{module_counter[module_type]}"

        # Параметры слоя
        num_params = sum(p.numel() for p in module.parameters())

        # Размеры (оцениваем)
        if isinstance(module, nn.Conv2d):
            channels = module.out_channels
            size_type = 'Conv'
            details = f"k={module.kernel_size}, stride={module.stride}"
        elif isinstance(module, nn.Linear):
            channels = module.out_features
            size_type = 'Linear'
            details = f"in={module.in_features}, out={module.out_features}"
        elif isinstance(module, nn.BatchNorm2d):
            channels = module.num_features
            size_type = 'Norm'
            details = f"features={module.num_features}"
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            channels = 1
            size_type = 'Pool'
            details = f"k={module.kernel_size}"
        elif isinstance(module, nn.Dropout):
            channels = 1
            size_type = 'Dropout'
            details = f"p={module.p}"
        elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            channels = 1
            size_type = 'Activation'
            details = module_type
        else:
            channels = 1
            size_type = 'Other'
            details = module_type

        # Сохраняем информацию
        layers_data.append({
            'id': f"{path}.{name}" if path else name,
            'name': name,
            'type': module_type,
            'size_type': size_type,
            'details': details,
            'depth': depth,
            'params': num_params,
            'channels': channels,
            'has_children': len(list(module.children())) > 0
        })

        # Рекурсивно анализируем дочерние модули
        child_idx = 0
        for child_name, child_module in module.named_children():
            if not child_name:
                child_name = f"child_{child_idx}"
                child_idx += 1

            child_path = f"{path}.{name}" if path else name
            analyze_module(child_module, child_name, child_path, depth + 1)

    # Начинаем анализ
    analyze_module(model, model.__class__.__name__)

    print(f"Собрана информация о {len(layers_data)} модулях")

    # Подготовка данных для визуализации
    leaf_layers = [layer for layer in layers_data if not layer['has_children']]

    if not leaf_layers:
        print("Нет данных для визуализации")
        return None

    depths = [layer['depth'] for layer in leaf_layers]
    params = [np.log10(max(layer['params'], 1)) for layer in leaf_layers]
    channels = [np.log10(max(layer['channels'], 1)) for layer in leaf_layers]
    names = [layer['name'] for layer in leaf_layers]
    types = [layer['type'] for layer in leaf_layers]
    details = [layer['details'] for layer in leaf_layers]

    # Цвета по типам слоев
    unique_types = list(set(types))
    colorscale = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                  '#A2D9CE', '#F9E79F', '#D7BDE2', '#A9CCE3', '#FAD7A0']

    type_to_color = {}
    for i, t in enumerate(unique_types):
        type_to_color[t] = colorscale[i % len(colorscale)]

    colors = [type_to_color[t] for t in types]

    # Создаем фигуру
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
               [{'type': 'treemap'}, {'type': 'bar'}]],
        subplot_titles=('3D Архитектура', 'Зависимость параметров от глубины',
                        'Иерархия параметров', 'Распределение типов слоев'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # 1. 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=depths,
            y=params,
            z=channels,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey'),
                symbol='circle'
            ),
            text=[f"<b>{n}</b><br>Тип: {t}<br>Параметры: {leaf_layers[i]['params']:,}<br>Детали: {d}"
                  for i, (n, t, d) in enumerate(zip(names, types, details))],
            hoverinfo='text',
            name='Слои'
        ),
        row=1, col=1
    )

    # 2. 2D scatter plot (параметры vs глубина)
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=[layer['params'] for layer in leaf_layers],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.7
            ),
            text=names,
            hovertext=[f"{t}: {layer['params']:,} params" for t, layer in zip(types, leaf_layers)],
            hoverinfo='text',
            name='Параметры'
        ),
        row=1, col=2
    )

    # 3. Treemap для иерархии параметров
    # Создаем иерархическую структуру
    labels = []
    parents = []
    values = []
    custom_data = []

    # Добавляем корневой узел
    root_name = model.__class__.__name__
    labels.append(root_name)
    parents.append("")
    values.append(sum(layer['params'] for layer in leaf_layers))
    custom_data.append(["Root", "All parameters"])

    # Добавляем все листовые узлы
    for layer in leaf_layers:
        if layer['params'] > 0:  # Только слои с параметрами
            labels.append(layer['name'])
            parents.append(root_name)
            values.append(layer['params'])
            custom_data.append([layer['type'], layer['details']])

    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=[type_to_color.get(layer['type'], '#CCCCCC')
                        for layer in leaf_layers] if leaf_layers else ['#CCCCCC']
            ),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Параметров: %{value:,}<br>Тип: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            customdata=custom_data,
            name="Параметры"
        ),
        row=2, col=1
    )

    # 4. Bar chart распределения типов слоев
    type_counts = defaultdict(int)
    type_params = defaultdict(int)

    for layer in leaf_layers:
        type_counts[layer['type']] += 1
        type_params[layer['type']] += layer['params']

    fig.add_trace(
        go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            text=[f"{type_params[t]:,}" for t in type_counts.keys()],
            textposition='auto',
            marker_color=[type_to_color.get(t, '#CCCCCC') for t in type_counts.keys()],
            name="Количество слоев",
            hovertemplate="<b>%{x}</b><br>Количество: %{y}<br>Всего параметров: %{text}<extra></extra>"
        ),
        row=2, col=2
    )

    # Обновляем layout
    fig.update_layout(
        title_text=f"Визуализация модели: {model.__class__.__name__}",
        height=900,
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Обновляем оси
    fig.update_xaxes(title_text="Глубина", row=1, col=1)
    fig.update_yaxes(title_text="log10(Параметры)", row=1, col=1)
    fig.update_zaxes(title_text="log10(Каналы)", row=1, col=1)

    fig.update_xaxes(title_text="Глубина слоя", row=1, col=2)
    fig.update_yaxes(title_text="Количество параметров", type="log", row=1, col=2)

    fig.update_xaxes(title_text="Типы слоев", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Количество", row=2, col=2)

    # Настройка сцены для 3D
    fig.update_scenes(
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        row=1, col=1
    )

    # Сохраняем и отображаем
    try:
        fig.write_html("model_visualization.html")
        print("Визуализация сохранена как model_visualization.html")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")

    fig.show()

    # Выводим текстовую сводку
    print("\n" + "=" * 60)
    print("ТЕКСТОВАЯ СВОДКА ПО МОДЕЛИ")
    print("=" * 60)

    total_params = sum(layer['params'] for layer in leaf_layers)
    total_layers = len(leaf_layers)

    print(f"Всего слоев: {total_layers}")
    print(f"Всего параметров: {total_params:,}")
    print(f"Слои с параметрами: {sum(1 for l in leaf_layers if l['params'] > 0)}")
    print(f"Максимальная глубина: {max(depths) if depths else 0}")

    # Топ-5 слоев по количеству параметров
    print("\nТоп-5 слоев по количеству параметров:")
    sorted_by_params = sorted(leaf_layers, key=lambda x: x['params'], reverse=True)[:5]
    for i, layer in enumerate(sorted_by_params, 1):
        print(f"  {i}. {layer['name']} ({layer['type']}): {layer['params']:,} params")

    return fig, layers_data


# Альтернативная упрощенная версия без рекурсии
def visualize_model_simple(model):
    """
    Упрощенная визуализация без сложной рекурсии
    """
    # Собираем плоский список всех модулей
    modules = []

    for name, module in model.named_modules():
        # Пропускаем корневой модуль, если у него нет параметров
        if name == '':
            continue

        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:  # Только модули с параметрами
            modules.append({
                'name': name,
                'type': module.__class__.__name__,
                'params': num_params,
                'depth': name.count('.')
            })

    if not modules:
        print("Модель не содержит параметров")
        return

    # Создаем визуализацию
    fig = go.Figure()

    # Подготовка данных
    names = [m['name'] for m in modules]
    params = [m['params'] for m in modules]
    depths = [m['depth'] for m in modules]
    types = [m['type'] for m in modules]

    # Группируем по типам для цветов
    unique_types = list(set(types))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}

    # Создаем график
    fig.add_trace(go.Bar(
        x=names,
        y=params,
        marker_color=[color_map[t] for t in types],
        text=[f"{p:,}" for p in params],
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Тип: %{customdata}<br>Параметров: %{y:,}<extra></extra>",
        customdata=types
    ))

    fig.update_layout(
        title=f"Параметры модели: {model.__class__.__name__}",
        xaxis_title="Слои",
        yaxis_title="Количество параметров (log scale)",
        yaxis_type="log",
        xaxis_tickangle=45,
        height=600,
        showlegend=False
    )

    # Добавляем аннотацию с общей статистикой
    total_params = sum(params)
    fig.add_annotation(
        x=0.5, y=0.95,
        xref="paper", yref="paper",
        text=f"Всего параметров: {total_params:,}<br>Всего слоев: {len(modules)}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.show()

    # Выводим статистику
    print(f"\nСтатистика модели {model.__class__.__name__}:")
    print(f"  Всего слоев с параметрами: {len(modules)}")
    print(f"  Всего параметров: {total_params:,}")

    # Группировка по типам
    type_stats = {}
    for module in modules:
        t = module['type']
        if t not in type_stats:
            type_stats[t] = {'count': 0, 'params': 0}
        type_stats[t]['count'] += 1
        type_stats[t]['params'] += module['params']

    print("\nПо типам слоев:")
    for t, stats in sorted(type_stats.items(), key=lambda x: x[1]['params'], reverse=True):
        print(f"  {t}: {stats['count']} слоев, {stats['params']:,} параметров")

    return fig


# Пример использования с обработкой ошибок
if __name__ == "__main__":
    # Создаем тестовую модель
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 56 * 56, 256),  # Предполагаем вход 224x224
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x


    try:
        model = SimpleCNN()

        print("1. Пробуем полную интерактивную визуализацию...")
        try:
            fig, data = visualize_model_interactive(model)
            print("✓ Успешно!")
        except RecursionError:
            print("✗ Ошибка рекурсии, используем упрощенную версию...")
            fig = visualize_model_simple(model)
        except Exception as e:
            print(f"✗ Другая ошибка: {e}")
            print("Пробуем упрощенную версию...")
            fig = visualize_model_simple(model)

        print("\n2. Используем упрощенную версию как резерв...")
        fig_simple = visualize_model_simple(model)

    except Exception as e:
        print(f"Критическая ошибка: {e}")


if __name__ == '__main__':
    model = FomoModelResV1()
    visualize_model_interactive(model)