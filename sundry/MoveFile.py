import os, shutil


def movefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


if __name__ == '__main__':
    for i in os.listdir('../GlobalEnvironmentChange/FinalWork/cache/cache/data'):
        srcdir = '../GlobalEnvironmentChange/FinalWork/cache/data/' + i
        filelist = os.listdir(srcdir)
        if 'download.json' in filelist:
            filelist.remove('download.json')
        dstdir = 'E:/DataBackup/sounding_station/' + i
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        for j in filelist:
            movefile(srcdir + '/' + j, dstdir + '/' + j)
