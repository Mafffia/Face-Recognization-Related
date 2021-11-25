import os
def get_img_pairs_list(pairs_txt_path,img_path):
    """ 指定图片组合及其所在文件，返回各图片对的绝对路径
        Args:
            pairs_txt_path：图片pairs文件，里面是6000对图片名字的组合
            img_path：图片所在文件夹
        return:
            img_pairs_list：深度为2的list，每一个二级list存放的是一对图片的绝对路径
    """
    file = open(pairs_txt_path)
    print(file)
    img_pairs_list,labels = [],[]
    for line in file.readlines():
        img_pairs = []
        line = line.replace('\n','')
        print(line)
        if(line==''):
            continue
        line_list = line.split('\t')
        if len(line_list) == 3:
            # 图片路径示例：
            # 'C:\Users\thinkpad1\Desktop\image_set\lfw_funneled\Tina_Fey\Tina_Fey_0001.jpg'
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[1])[-4:]+'.jpg')
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[2])[-4:]+'.jpg')
            labels.append(1)
        elif len(line_list) == 4:
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[1])[-4:]+'.jpg')
            img_pairs.append(img_path+'\\'+line_list[2]+'\\'+line_list[2]+'_'+('000'+line_list[3])[-4:]+'.jpg')
            labels.append(0)
        else:
            continue
        
        img_pairs_list.append(img_pairs)
    return img_pairs_list,labels


def gen_all_pic_labels(img_path):
    pics = []
    names = []
    genders = []
    gender = gender_dict()
    for i in os.listdir(img_path):
        if os.path.isdir(os.path.join(img_path,i)):
            for filename in os.listdir(os.path.join(img_path,i)):
                if(filename=='.DS_Store'):
                    continue
                path = os.path.join(img_path,i,filename)
                if(filename in gender.keys()):
                    genders.append(gender[filename])
                else:
                    print(filename)
                pics.append(str(path))
                names.append(i)
    return pics,names,genders

def gender_dict():
    gender = {}
    with open('female_names.txt','r') as f:
        list = f.read().splitlines()
        for line in list:
            gender[str(line)] = 'female'

    with open('male_names.txt','r') as f:
        list = f.read().splitlines()
        for line in list:
            gender[str(line)] = 'male'
    return gender




# imgpair_label,labels = get_img_pairs_list('./lfw_funneled/pairs.txt','./lfw_funneled')
# # print(imgpair_label)
# print(len(imgpair_label),len(labels))
# print(imgpair_label[0],labels[0])


pics,names,genders = gen_all_pic_labels('./lfw_funneled/')
# print(pics[0],names[0])