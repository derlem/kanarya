import os


def get_file_list(src_fol, ext):
    f_list = []
    for subdir, _, files in os.walk(src_fol):
        for filename in files:
            _, f_ext = os.path.splitext(filename)
            if f_ext == ext:
                full_file_path = os.path.join(subdir, filename)
                f_list.append(full_file_path)
    return f_list


def get_eq_file_list(src_fol, trgt_fol, in_ext, out_ext):
    f_src_list = []
    f_trgt_list = []
    for subdir, dirs, files in os.walk(src_fol):
        for filename in files:
            if not filename.endswith(in_ext):
                continue
            # create subdir for writing
            sub_fol = subdir.split("/")[-1]
            trgt_subfol = os.path.join(trgt_fol, sub_fol)

            if not os.path.exists(trgt_subfol):
                os.makedirs(trgt_subfol)

            src_file = os.path.join(subdir, filename)
            f_src_list.append(src_file)
            filename = filename.replace(in_ext, out_ext)

            # make the path
            trgt_file = os.path.join(trgt_subfol, filename)
            f_trgt_list.append(trgt_file)
    return f_src_list, f_trgt_list


