# model = tmp_queue.get()
# try:
#     with Path.open(Path(model), "r") as file:
#         code = file.read()
#         print(code)
#         exec(compile(file.read(), model, 'exec'))
#     self.queue.put(model)
#     print(Path(model).name + "\033[95m运行成功\033[0m")
# except:
#     print(Path(model).name + "\033[94m运行失败\033[0m")
#     with Path.open(self.log_file / (self.model_name + int(time.time()).__str__() + ".txt"), "a+",
#                    encoding="utf8") as log_file:
#         traceback.print_exc(file=log_file)
