import os
import csv
import copy
import numpy as np
import shutil
import string


def get_voclexb_labels(voice_list, face_list, celeb_ids):
    """
    合并voice和face中的同类项目
    :param voice_list:
    :param face_list:
    :return:
    """
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names
    voice_face_list = []

    #  通过列表推导式 保留同类项
    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = list(names)
    label_dict = dict(zip(names, range(len(names))))
    temp1 = []
    temp2 = []

    # 建立


    # 建立联合组voice+face-list
    for name in names[:]:
        for item in voice_list:   # 为list增加序号label_id
            if name == item['name']:
                temp1.append(item['filepath'])
        for item in face_list:
            if name == item['name']:
                temp2.append(item['filepath'])
        voice_face_list.append({'name': name, 'id_num': label_dict[name], 'id': celeb_ids[name],
                               'voice_path': copy.deepcopy(temp1), 'face_path': copy.deepcopy(temp2)})
        print(name)
        temp1.clear()
        temp2.clear()

    return voice_face_list


def get_dataset_files(data_dir, data_ext, celeb_ids):
    """
    从文件夹中寻找所有voice和face数据
    """
    data_list = []

    # read data directory
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder)   #  default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
                data_list.append({'name': celeb_name,  'filepath': filepath })
    return data_list


def get_voclexb_csv(csv_files, voice_folder, face_folder):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    csv_headers = ['name','id_num', 'id' ,'voice_path', 'face_path']
    triplet_list = []
    actor_dict, actor_dict1 = {}, {}

    with open(csv_files) as f:
        lines = f.readlines()[1:]
        for line in lines:
            actor_ID, name, gender, nation, _ = line.rstrip("\n").split('\t')
            actor_dict[actor_ID] = name
            actor_dict1[name] = actor_ID

    voice_list = get_dataset_files(voice_folder, 'wav', actor_dict)
    face_list = get_dataset_files(face_folder, 'jpg', actor_dict)
    voice_face_list = get_voclexb_labels(voice_list, face_list, actor_dict1)

    # np.save('./dataset/voclexb-VGG_face-datasets/voice_face_list.npy', voice_face_list)
    csv_pth = os.path.join('../dataset/voclexb-VGG_face-datasets/', 'voice_face_list.csv')
    with open(csv_pth,'w',newline='', ) as f:
        f_scv = csv.DictWriter(f,csv_headers,delimiter = '\t', lineterminator = '\n')
        f_scv.writeheader()
        f_scv.writerows(voice_face_list)

    return voice_face_list, len(actor_dict)


if __name__ == '__main__':
    # get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/fbank'

    csv_files = './dataset/voclexb-VGG_face-datasets/vox1_meta.csv'
    voice_folder = './dataset/voclexb-VGG_face-datasets/2-voice-wav'
    face_folder = './dataset/voclexb-VGG_face-datasets/1-face'
    list, num = get_voclexb_csv(csv_files, voice_folder, face_folder)
    pass


    # voice_data_pth = './datasets/RAVDESS/2 wave-Actor1-24-32k'
    # image_data_pth = './datasets/RAVDESS/1 image-Actor1-24-single'
    # csv_pth = "./datasets/RAVDESS"
    # get_RAVDESS_csv(voice_data_pth, image_data_pth, csv_pth, 'voice')


