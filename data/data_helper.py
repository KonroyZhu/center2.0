import os

def raw_data(path,enc="utf-8"):
    if not path.endswith("/"): path=path+"/"
    path_li=[path+c for c in (os.listdir(path))]
    raw_dt=[]
    for p in path_li:
        sen_li=[l.replace("\n","").split("#") for l in open(p,encoding=enc).readlines()]
        raw_dt=raw_dt+sen_li
    return raw_dt

def _2filted(raw_data):
    # 将中心句标为2的(两句均为中心句)去掉，兼容后续标注的数据集
    filted=[]
    for dt in raw_data:
        if dt[-1] != "2":
            filted.append(dt)
    return filted

raw_dt=_2filted(raw_data("segment"))
