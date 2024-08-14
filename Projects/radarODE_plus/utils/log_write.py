import os



def create_log(root_path, loss_array, loss_tyoe):
    loss_file_path = os.path.join(root_path, loss_tyoe+'.txt')
    with open(loss_file_path, 'w', encoding='utf8') as f:
        for item in loss_array:
            f.write(str(item))
            f.write('\n')
