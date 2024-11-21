import os
import pdb

refer_path = [
        "./data_path/Train/라벨링데이터/",
        "./data_path/Test/라벨링데이터/",
        "./data_path/Validation/라벨링데이터/",
]
docs_names = set()

def navigate_directory(base_path, cur_relative_path, path_collection):
    cur_path = base_path + '/' + cur_relative_path
    sub_dirs = os.listdir(cur_path)
    sub_dirs = [x for x in sub_dirs if os.path.isdir(cur_path + '/' + x)]
    if not sub_dirs:
        return path_collection + [cur_relative_path]
    for d in sub_dirs:
        path_collection = navigate_directory(
            base_path,
            cur_relative_path + '/' + d,
            path_collection)
    return path_collection

if __name__ == '__main__':
    all_paths = []
    for p in refer_path:
        tmp_paths = navigate_directory(p, '', [])
        all_paths += tmp_paths
    pdb.set_trace()
