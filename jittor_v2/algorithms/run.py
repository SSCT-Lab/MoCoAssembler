import shutil
import traceback
from importlib import import_module
from datetime import datetime
import file_paths
import os
from filter import Filter


def run_single_model(model_name: str, model_type: str) -> bool:
    flag = True
    error = ''
    try:
        module_name = model_name.replace('.py', '')
        module = import_module(module_name)
        module.go()
    except Exception as e:
        flag = False
        error = error + str(traceback.format_exc())

    if flag:
        # print(model_name + ' succeed')
        return True
    else:
        f = Filter()
        if not f.judge(error):
            # print(model_name + ' has error, but filtered out...')
            return False
        print(model_name + ' has error, and witten into log...')
        result = model_name + '  error   :          \n'
        now = datetime.now()
        timenow = str(now.year) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "   " + str(
            now.hour).zfill(2) + ':' + str(now.minute).zfill(2)
        result = result + 'Time:  ' + timenow.replace(' ', '-').replace(':', '-') + '\n'
        result = result + 'Info: \n' + error + '\n\n\n'
        with open(os.path.join(file_paths.MAIN_PATH, 'logs', model_type, 'log.txt'), 'a') as f:
            f.write(result)

        source = os.path.join(file_paths.MUTATED_MODEL_PATH, model_type, model_name)
        target = os.path.join(file_paths.SAVED_MODEL_PATH, model_type)
        new_source = source[:-3] + '_' + timenow.replace(' ', '-').replace(':', '-') + '.py'
        try:
            os.rename(source, new_source)
            shutil.copy(new_source, target)
        except Exception as e:
            print(e)
        return False



if __name__ == '__main__':
    run_single_model('testnet-1-1.py')
