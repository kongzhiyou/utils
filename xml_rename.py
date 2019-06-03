
import os
import glob

'''
使用labelImg标注时忘记将图片分布在几个目录中使用该方法可以对 .jgp,.xml进行重命名
'''

root_path = '/Users/peter/Desktop/video'

dir_list = glob.glob(root_path+'/*')

for dir in dir_list:
    dir_name = dir.split('/')[-1]
    file_list = glob.glob(dir+'/*')
    for dfile in file_list:
        file_name = dfile.split('/')[-1]
        file_suf = file_name.split('.')[-1]
        if file_suf=='xml':
            os.rename(dfile,dir+'/'+dir_name+'_'+file_name.split('.')[0]+'.xml')
        elif file_suf=='jpg':
            os.rename(dfile, dir + '/' + dir_name + '_' + file_name.split('.')[0] + '.jpg')
