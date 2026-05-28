
# LLM-BotNet-detection
**Обнаружение ботнет-атак в IoT-трафике с помощью LLM-генерируемого кода**

## Цель и задачи 

**Цель**: исследовать возможность построения рабочей системы (Python-приложения) обнаружения ботнет-атак посредством промптов для LLM.

**Основной функционал итоговой системы:** автоэнкодер, который обучался на нормальном трафике, способен обнаружить аномальный (вредоносный) трафик в сетях IoT. Если декодировщик ошибка восстановления данных, пришедших на вход кодировщику, превышает порог, то трафик классифицируется как атака.

**Задачи:**
1. Разработать промпт для генерации Python-кода.
2. Получить Python-код через LLM: Claude Opus 4.6, DeepSeek V4, GigaChat.
3. Сравнить результаты на EN и RU промптах.
4. Сопоставить LLM-код с "ручными" моделями (автоэнкодер, Isolation Forest, One-Class SVM).
5. Оценить результаты на всех IoT-устройствах датасета.


## Датасет
 
**N-BaIoT** — реальный сетевой трафик с 9 коммерческих IoT-устройств (камеры, термостаты, дверные звонки), заражённых ботнетами Mirai и BASHLITE. Каждая запись содержит 115 числовых признаков — статистик сетевого потока (ПРИМЕРЫ).
 
- **Источник:** [Kaggle](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) / [UCI ML Repository, ID 442](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)
- **Оригинальная статья:** Meidan et al. (2018) "N-BaIoT — Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders"
| ID | Устройство | Тип |
|---|---|---|
| 1 | Danmini Doorbell | Дверной звонок |
| 2 | Ecobee Thermostat | Термостат |
| 3 | Ennio Doorbell | Дверной звонок |
| 4 | Philips B120N10 Baby Monitor | Видеоняня |
| 5 | Provision PT-737E | Камера безопасности |
| 6 | Provision PT-838 | Камера безопасности |
| 7 | Samsung SNH-1011 Webcam | Веб-камера |
| 8 | SimpleHome XCS7-1002-WHT | Камера безопасности |
| 9 | SimpleHome XCS7-1003-WHT | Камера безопасности |


## Как формировался промпт

Для получения целевых промптов, генерирующих наше Python-приложения, мы предварительно написали промпт для их получения -- мета промпт.
Он был сгенерирован нейтральной (не используемой в дальнейших экспериментах) моделью Codex / GPT 5.5.

### Цепочка генерации
 
```mermaid
graph LR
    A[Методология<br>Monge Martinez 2023] --> B[Мета-промпт]
    B --> C[Codex / GPT 5.5]
    C --> D[Промпт EN]
    C --> E[Промпт RU]
    D --> F[Claude Opus 4.6]
    D --> G[DeepSeek V4]
    D --> H[GigaChat]
    E --> F
    E --> G
    E --> H
    F --> I[6 скриптов]
    G --> I
    H --> I
    I --> J[Сравнение<br>результатов]
    K[Бейзлайн<br>руками] --> J
```

### 4-компонентная структура промпта
 
На основе методологии из Monge Martinez (2023) ["Using LLMs and GPT to streamline data analysis in cybersecurity incidents"](https://luckyluk3.medium.com/using-llms-and-gpt-to-streamline-data-analysis-in-cybersecurity-incidents-ebeb0d23e01b):
 
| Компонент | Содержание |
|---|---|
| Роль| Генератор Python-кода для кибербезопасности |
| Контекст| Обнаружение аномалий автоэнкодером, подход N-BaIoT |
| Данные | Описание CSV-файлов, 115 признаков, структура именования |
| Шаги | Загрузка → предобработка → модель → порог → оценка → визуализация |
 
Мета-промпт подан в Codex (GPT 5.5), который не участвует в основном эксперименте, что исключает предвзятость.


## Выбор LLM
 
| Модель | Происхождение | Почему выбрана |
|---|---|---|
| **Claude Opus 4.6** | США (Anthropic) | Лидер по генерации кода (SWE-Bench 80.8%) |
| **DeepSeek V4** | Китай (DeepSeek) | Сильная в коде (лучше справляется с EN промптами), бюджетнее Claude |
| **GigaChat** | Россия (Сбер) | Русскоязычная модель — для проверки RU промпта |
 
Каждая модель получила оба промпта (EN и RU) в отдельных чатах → 6 скриптов.

## Бейзлайн «руками» (???)
 
Для сравнения с LLM-генерированным кодом реализованы три метода вручную:
 
| Метод | Зачем |
|---|---|
| **Автоэнкодер** (Dropout + ReLU + StandardScaler) | Та же идея, но другие архитектурные решения — показывает, насколько выбор LLM (BatchNorm, LeakyReLU, MinMaxScaler) влияет на результат |
| **Isolation Forest** | Классический метод обнаружения аномалий без нейросетей — нужен ли вообще автоэнкодер? |
| **One-Class SVM** | Другой принцип (граница в пространстве признаков) — третья точка сравнения |

## Структура репозитория
 
```
├── README.md
├── meta_prompt_for_codex.md          # Мета-промпт для Codex
├── prompts/
│   ├── prompt_en.md                  # Промпт EN (сгенерирован Codex)
│   └── prompt_ru.md                  # Промпт RU (сгенерирован Codex)
├── generated_code/
│   ├── claude_en.py                  # Claude Opus 4.6 + EN промпт
│   ├── claude_ru.py                  # Claude Opus 4.6 + RU промпт
│   ├── deepseek_en.py                # DeepSeek V4 + EN промпт
│   ├── deepseek_ru.py                # DeepSeek V4 + RU промпт
│   ├── gigachat_en.py                # GigaChat + EN промпт
│   └── gigachat_ru.py                # GigaChat + RU промпт
├── baseline/
│   └── botnet_detection_manual.py    # Ручной бейзлайн (AE + IsoForest + OC-SVM)
├── results/
│   ├── device_1_danmini_doorbell/    # Результаты по устройству 1
│   ├── device_2_ecobee_thermostat/
│   ├── ...
│   └── device_9_simplehome_1003/
├── run_all.sh                        # Скрипт для прогона всех экспериментов (9 устройств)
├── requirements.txt
├── presentation/                     # Слайды (LaTeX)
└── diagram1.png, diagram2.png        # Схемы эксперимента
```

## Как запустить
### 1. Установка зависимостей
 
```bash
pip3 install -r requirements.txt
```

### 2. Скачивание датасета
Скачать CSV-файлы с [Kaggle](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) и распаковать в папку `dataset/`.
 
### 3. Запуск одного скрипта (только на устройстве 1 -- пример)
 
```bash
cd generated_code
python3 claude_en.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/device_1_danmini_doorbell/claude_en.txt
```

### 4. Запуск всех экспериментов (все 6 скриптов × 9 устройств)
 
```bash
chmod +x run_all.sh
./run_all.sh
```


## Предварительные наблюдения

**Объём сгенерированного кода:**
 
| Модель | EN (строк) | RU (строк) |
|---|---|---|
| Claude Opus 4.6 | 416 | 482 |
| DeepSeek V4 | 387 | 430 |
| GigaChat | 258 | 211 |
 
- Claude — самый подробный код с детальными комментариями и именованными слоями
- DeepSeek — хороший баланс между полнотой и компактностью
- GigaChat — самый лаконичный, но все компоненты на месте

**Обнаруженные ошибки:**
- `deepseek_ru.py` содержал 2 бага: некорректное чтение CSV (`header=None` вместо автоопределения заголовков) и несуществующая переменная `history` в возврате функции. Скрипты Claude и GigaChat этих проблем не имели.






## ручные методы

## codex для мета промпта
