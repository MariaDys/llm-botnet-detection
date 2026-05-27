
# llm-botnet-detection

Claude Opus 4.6: лидер по генерации кода

DeepSeek V4/R1: сильная в коде, сильнее на английском, гораздо бюджетнее чем клод

GigaChat: русскоязычная модель – русский промпт

## датасет 
https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset

## ручные методы



## прогоны -- вывод сохраняем в папку results 
сначала установить requirements.txt через `pip3 install -r requirements.txt`

python3 claude_en.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/claude_en.txt
python3 claude_ru.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/claude_ru.txt
python3 deepseek_en.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/deepseek_en.txt
python3 deepseek_ru.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/deepseek_ru.txt

«Скрипт deepseek_ru.py потребовал ручного исправления ошибки загрузки данных (заголовки CSV интерпретировались как числовые значения). Скрипты Claude и GigaChat этой проблемы не имели

python3 gigachat_en.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/gigachat_en.txt
python3 gigachat_ru.py --data_dir ../dataset --device_id 1 2>&1 | tee ../results/gigachat_ru.txt

## скрипт для прогона 6 скриптов по всем 9 девайсам с сохранением результатов в папки

run_all.sh

`chmod +x run_all.sh`
`./run_all.sh`

## codex для мета промпта
