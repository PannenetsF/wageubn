def _isinstance(obj, cls_list):
    flag_list = [isinstance(obj, cls) for cls in cls_list]
    flag = False
    for x in flag_list:
        flag = x or flag
    return flag
