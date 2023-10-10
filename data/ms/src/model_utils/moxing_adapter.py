# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Moxing adapter for ModelArts"""


import os
import functools
import time
import json

from .config import config

import mindspore as ms

if config.enable_modelarts:
    import moxing as mox


_global_sync_count = 0


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    job_id = os.getenv('JOB_ID')
    job_id = job_id if job_id != "" else "default"
    return job_id


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path, threads=128)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def moxing_wrapper(pre_process=None, post_process=None):
    """
    Moxing wrapper to download dataset and upload outputs.
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            # Download data from data_url
            if config.enable_modelarts:
                if config.multi_data_url:
                    multi_data_url = json.loads(config.multi_data_url)
                    for data_url in multi_data_url:
                        sync_data(data_url["dataset_url"], config.data_path)
                        print("Dataset downloaded: ", os.listdir(config.data_path))
                elif config.data_url:
                    sync_data(config.data_url, config.data_path)
                    print("Dataset downloaded: ", os.listdir(config.data_path))
                else:
                    raise ValueError
                if config.ckpt_url and config.ckpt_url != "[]":
                    sync_data(config.ckpt_url, config.load_path)
                    print("Preload downloaded: ", config.load_path)
                if config.train_url:
                    sync_data(config.train_url, config.output_path)
                    print("Workspace downloaded: ", os.listdir(config.output_path))

                ms.context.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                if not os.path.exists(config.output_path):
                    os.makedirs(config.output_path)

                if pre_process:
                    pre_process()

            if config.enable_profiling:
                profiler = ms.profiler.Profiler()

            run_func(*args, **kwargs)

            if config.enable_profiling:
                profiler.analyse()

            # Upload data to train_url
            if config.enable_modelarts:
                if post_process:
                    post_process()

                if config.train_url:
                    print("Start to copy output directory")
                    sync_data(config.output_path, config.train_url)
                    sync_data('/tmp/log', config.train_url)
                if config.result_url:
                    if config.result_path:
                        print("Start to copy result directory")
                        sync_data(config.result_path, config.result_url)
                    sync_data('/tmp/log', config.result_url)
        return wrapped_func
    return wrapper


def modelarts_process():
    """ modelarts process """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

    config.dataset_path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    # if config.pretrain_ckpt:
    #     config.pretrain_ckpt = os.path.join(config.output_path, config.pretrain_ckpt)
