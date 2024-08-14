import os
import datetime


def create_folder(flag):
    root = 'Model_saved'

    # 获取当前时间
    current_time = datetime.datetime.now()

    # 格式化时间为年月日时分秒
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # 要创建的文件夹名称
    folder_name = f"folder_{flag}"

    if not os.path.exists(os.path.join(root, folder_name)):
        # 创建文件夹
        os.mkdir(os.path.join(root, folder_name))

    return os.path.join(root, folder_name)

if __name__ == '__main__':
    create_folder()