import os

# Список папок, которые мы КАТЕГОРИЧЕСКИ игнорируем
EXCLUDE_DIRS = {
    'rag_venv', 'venv', '.venv', 'env',  # Виртуальные окружения
    '__pycache__', '.git', '.idea', '.vscode', # Служебные папки IDE и Git
    'node_modules', 'dist', 'build', 'site-packages' # Библиотеки и сборки
}

OUTPUT_FILE = 'my_project_code.txt'

def collect_python_files():
    root_dir = os.getcwd()
    collected_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Модифицируем dirs на месте, чтобы os.walk не заходил в исключенные папки
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                # Берем только .py файлы и игнорируем сам скрипт сборщика
                if file.endswith('.py') and file != os.path.basename(__file__):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    # Дополнительная проверка: если в пути есть site-packages, пропускаем
                    if 'site-packages' in relative_path:
                        continue

                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"FILE: {relative_path}\n")
                    outfile.write(f"{'='*50}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                        collected_count += 1
                        print(f"Добавлен: {relative_path}")
                    except Exception as e:
                        outfile.write(f"// Ошибка чтения файла: {e}\n")
                    
                    outfile.write("\n\n")

    print(f"\nГотово! Теперь в файле только ваш код: {OUTPUT_FILE}")
    print(f"Всего обработано файлов: {collected_count}")

if __name__ == "__main__":
    collect_python_files()