import os

def raw_data(path,enc="utf-8"):
    if not path.endswith("/"): path=path+"/"
    path_li=[path+c for c in (os.listdir(path))]
    raw_dt=[]
    for p in path_li:
        sen_li=[l.replace("\n","").split("#") for l in open(p,encoding=enc).readlines()]
        raw_dt=raw_dt+sen_li
    return raw_dt

raw_dt=raw_data("./segment")
for dt in raw_dt:
    print(dt)