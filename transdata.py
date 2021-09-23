from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
 
#the path you want to save your results for coco to voc
savepath="/home/pascal/pascal/darknet/data/transdata/train20/"  #保存提取类的路径,我放在同一路径下
img_dir=savepath+'images/'
anno_dir=savepath+'Annotations/'
datasets_list=['train2014', 'val2014']
#datasets_list=['train2014']
 
classes_names = ['cell phone']  #coco有80类，这里写要提取类的名字，以person为例
#Store annotations and train2014/val2014/... in this folder
dataDir= '/home/pascal/pascal/darknet/data/coco/'  #原coco数据集
ana_txt_save_path = "/home/pascal/pascal/darknet/data/transdata/trainph/"
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
 
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
mkr(img_dir)
mkr(anno_dir)
def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(img,coco,dataset,filename,objs):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+filename[:-3]+'xml'
    img_path=dataDir+dataset+'/'+filename
    print(img_path)
    dst_imgpath=img_dir+filename
    try:
        img=cv2.imread(img_path)

    #if (img.shape[2] == 1):
        #print(filename + " not a RGB image")
        #return
        shutil.copy(img_path, dst_imgpath)
 
        head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
        tail = tailstr
        write_xml(anno_path,head, objs, tail)
    except:
        return
def savetxt(img,filename,dataset):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    ana_txt_name = filename.split(".")[0] + ".txt"
    print(ana_txt_name)
    img_path=dataDir+dataset+'/'+filename
    try:
        I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        for ann in anns:
            if ann['image_id']==img_id:
                box = convert((img_width,img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n"%(ann["category_id"], box[0], box[1], box[2], box[3]))
        f_txt.close()
    except:
        return


 
def showimg(coco,dataset,img,classes,cls_id,show=False):
    global dataDir
    global objs
    try:
        I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))


        #通过id，得到注释的信息
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
        # print(annIds)
        anns = coco.loadAnns(annIds)
        # print(anns)
        # coco.showAnns(anns)
        objs = []
        for ann in anns:
            class_name=classes[ann['category_id']]
            if class_name in classes_names:
                #print(class_name)
                if 'bbox' in ann:
                    bbox=ann['bbox']
                    xmin = int(bbox[0])
                    ymin = int(bbox[1])
                    xmax = int(bbox[2] + bbox[0])
                    ymax = int(bbox[3] + bbox[1])
                    obj = [class_name, xmin, ymin, xmax, ymax]
                    objs.append(obj)
                    draw = ImageDraw.Draw(I)
                    draw.rectangle([xmin, ymin, xmax, ymax])
    except:
        obj=[]
        
    
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
 
    return objs
 
for dataset in datasets_list:
    #./COCO/annotations/instances_train2014.json
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
 
    #COCO API for initializing annotated data
    coco = COCO(annFile)

    #show all classes in coco
    classes = id2name(coco)
    #print(classes)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    #print(classes_ids)
    for cls in classes_names:
        #Get ID number of this class
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        #print(cls,len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
            print(objs)
            if len(objs)!=0:
                save_annotations_and_imgs(img,coco, dataset, filename, objs)
                savetxt(img,filename,dataset)
