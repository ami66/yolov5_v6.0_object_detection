import cv2
from tqdm import tqdm
from detect_image_only import load_model,predict

from lxml.etree import Element, SubElement, tostring

def create_xml(list_xml,list_images,xml_path):
    """
    创建xml文件，依次写入xml文件必备关键字
    :param list_xml:   txt文件中的box
    :param list_images:   图片信息，xml中需要写入WHC
    :return:
    """
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(list_images[3])
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(list_images[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(list_images[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(list_images[2])

    if len(list_xml)>=1:        # 循环写入box
        for list_ in list_xml:
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            # if str(list_[4]) == "person":                # 根据条件筛选需要标注的标签,例如这里只标记person这类，不符合则直接跳过
            #     node_name.text = str(list_[4])
            # else:
            #     continue
            node_name.text = str(list_[4])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(list_[0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(list_[1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(list_[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(list_[3])

    xml = tostring(node_root, pretty_print=True)   # 格式化显示，该换行的换行

    file_name = list_images[3].split(".")[0]
    filename = xml_path+"/{}.xml".format(file_name)

    f = open(filename, "wb")
    f.write(xml)
    f.close()


if __name__ == '__main__':

    import os

    path = r"auto_label/images"   #待标注图片路径
    xml_path = r"auto_label/images"    #输出的xml标注文件保存路径
    yolo_model_weight='./weight/IDCard_v6x_best.pt'  #模型文件路径
    data_conf = './data/custom_data.yaml'  #数据集配置文件路径

    device='cpu'
    model, ini_model = load_model(device=device, weights=yolo_model_weight,data_conf = data_conf)
    for name in tqdm(os.listdir(path)):
        #print(name)
        xml_name=name.replace('.jpg','.xml')
        xml_p=os.path.join(path,xml_name)
        if not os.path.exists(xml_p):
            image = cv2.imread(os.path.join(path,name))
            try:
                list_image = (image.shape[0],image.shape[1],image.shape[2],name)             # 图片的宽高等信息
            except:
                continue

            xyxy_list=predict(image=image,model=model,ini_model=ini_model,device=device)       # img0检测后的图像，img_crop裁剪的图像
            if xyxy_list:
                create_xml(xyxy_list,list_image,xml_path)          # 生成标注的xml文件
            else:
                print('未检测到目标：',name)

        else:
            continue


