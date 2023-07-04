import os
import shutil

MAIN_PATH = os.path.join(os.getcwd())
RESULT_PATH = os.path.join(MAIN_PATH, 'results')
GENERATED_MODELS_PATH = os.path.join(RESULT_PATH, 'generated_models')
SAVED_MODELS_PATH = os.path.join(RESULT_PATH, 'saved_models')


def dfc(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"无法删除文件: {file_path}，错误信息: {e}")


if __name__ == '__main__':
    dfc(GENERATED_MODELS_PATH)
    dfc(SAVED_MODELS_PATH)
    open(os.path.join(RESULT_PATH, 'ed_log.txt'), 'w').close()
    open(os.path.join(RESULT_PATH, 'es_log.txt'), 'w').close()
    open(os.path.join(RESULT_PATH, 'os_log.txt'), 'w').close()
