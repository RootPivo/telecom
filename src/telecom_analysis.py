import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Создаем папку для сохранения графиков
if not os.path.exists('plots'):
    os.makedirs('plots')

# Функция для загрузки и обработки данных
def load_and_process_data(file_path):
    # Создаем папку data, если её нет
    data_dir = os.path.dirname(file_path) or '.'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Создано директория: {data_dir}")

    try:
        # Загружаем данные из локального файла
        data = pd.read_csv("C:/нужное/в/ОБЩАК/ML_уник/telecom/data/telecom_churn.csv")
        
        # Проверяем уникальные значения в 'Churn'
        print("Уникальные значения в 'Churn' до преобразования:", data['Churn'].unique())
        
        # Сначала заполняем пропуски (кроме 'Churn')
        if data.drop(columns=['Churn']).isnull().sum().any():
            print("Обнаружены пропуски в данных (кроме 'Churn'). Заполняем их нулями.")
            data[data.drop(columns=['Churn']).columns] = data[data.drop(columns=['Churn']).columns].fillna(0)
        
        # Преобразуем 'Churn' с обработкой NaN
        data['Churn'] = data['Churn'].map({'True': True, 'False': False})
        if data['Churn'].isna().any():
            print("Предупреждение: В 'Churn' есть значения, не соответствующие 'True' или 'False'. Заполняем их False.")
            data['Churn'] = data['Churn'].fillna(False)
        
        # Преобразуем другие столбцы
        for col in ['International plan', 'Voice mail plan']:
            data[col] = data[col].map({'Yes': True, 'No': False})
            if data[col].isna().any():
                print(f"Предупреждение: В '{col}' есть значения, не соответствующие 'Yes' или 'No'. Заполняем их False.")
                data[col] = data[col].fillna(False)
        
        return data
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None

# Основной анализ и визуализации
def analyze_data(data):
    # Дополнительная проверка на NaN в 'Customer service calls'
    if data['Customer service calls'].isna().any():
        print("Найдены NaN в 'Customer service calls'. Заполняем их нулями.")
        data['Customer service calls'].fillna(0, inplace=True)

    # 1. Уровень оттока
    churn_rate = data['Churn'].mean() * 100
    print(f"Уровень оттока: {churn_rate:.1f}%")

    # 2. Географический анализ
    state_churn = data.groupby('State')['Churn'].mean().sort_values(ascending=False) * 100
    top_states = state_churn.head(5)
    
    # 3. Использование услуг
    usage_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    charge_cols = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    data['Total minutes'] = data[usage_cols].sum(axis=1)
    data['Total charge'] = data[charge_cols].sum(axis=1)
    
    # 4. Служба поддержки
    service_calls_churn = data.groupby('Customer service calls')['Churn'].mean() * 100
    
    # 5. Корреляции
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numeric_cols].corr()

    # Применяем стиль Seaborn
    sns.set_theme(style='darkgrid')

    # Визуализации
    # Гистограмма уровня оттока
    plt.figure(figsize=(8, 6))
    plt.bar(['Отток', 'Оставшиеся'], [churn_rate, 100 - churn_rate], color=['#ff7300', '#82ca9d'])
    plt.title('Общий уровень оттока клиентов')
    plt.ylabel('Процент (%)')
    plt.savefig('plots/churn_rate.png')
    plt.close()

    # Круговая диаграмма оттока по штатам
    if not top_states.isna().any() and top_states.sum() > 0:
        plt.figure(figsize=(8, 6))
        plt.pie(top_states, labels=top_states.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title('Уровень оттока по топ-5 штатам')
        plt.savefig('plots/state_churn.png')
        plt.close()
    else:
        print("Предупреждение: Нельзя построить круговую диаграмму из-за нулевых или NaN значений.")

    # Диаграмма рассеяния (дневные минуты vs. счета)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total day minutes', y='Total day charge', hue='Churn', palette={True: '#ff7300', False: '#82ca9d'})
    plt.title('Дневные минуты vs. Дневные счета')
    plt.xlabel('Дневные минуты')
    plt.ylabel('Дневной счет ($)')
    plt.savefig('plots/minutes_vs_charge.png')
    plt.close()

    # Гистограмма оттока по обращениям в поддержку
    plt.figure(figsize=(10, 6))
    plt.bar(service_calls_churn.index, service_calls_churn, color='#8884d8')
    plt.title('Уровень оттока по количеству обращений в службу поддержки')
    plt.xlabel('Количество обращений')
    plt.ylabel('Уровень оттока (%)')
    plt.savefig('plots/service_calls_churn.png')
    plt.close()

    # Тепловая карта корреляций
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляции между числовыми признаками')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # Интересный факт
    low_churn_high_charge = data[(data['Churn'] == False) & (data['Total charge'] > data['Total charge'].quantile(0.75))]['State'].value_counts().idxmax()
    print(f"Интересный факт: Штат {low_churn_high_charge} имеет низкий отток, несмотря на высокие счета.")

    # Рекомендации
    recommendations = [
        "1. Конкурентные тарифы: Ввести скидки на дневные звонки или пакетные предложения для клиентов с высокими счетами (>$50), чтобы снизить отток.",
        f"2. Таргетированный маркетинг: Сфокусироваться на штатах с низким оттоком, таких как {low_churn_high_charge}, для рекламных кампаний.",
        "3. Улучшение службы поддержки: Внедрить тренинги для сотрудников, чтобы сократить количество обращений (>4), что значительно увеличивает отток.",
        "4. Персонализация тарифов: Продвигать голосовую почту для клиентов с высоким количеством сообщений и международные планы для активных пользователей международных звонков."
    ]
    return churn_rate, recommendations

# Основной блок
def main():
    print("Текущая рабочая директория:", os.getcwd())
    file_path = "C:/нужное/в/ОБЩАК/ML_уник/telecom/data/telecom_churn.csv"
    data = load_and_process_data(file_path)
    
    if data is None:
        print("Не удалось загрузить данные. Пожалуйста, проверьте файл или формат датасета.")
        return
    
    print("Данные успешно загружены. Начинаем анализ...")
    churn_rate, recommendations = analyze_data(data)
    
    # Вывод результатов
    print("\nРезультаты анализа:")
    print(f"Уровень оттока: {churn_rate:.1f}%")
    print("\nРекомендации для привлечения клиентов:")
    for rec in recommendations:
        print(rec)
    
    print("\nВизуализации сохранены в папке 'plots'.")

if __name__ == "__main__":
    main()