import os
import csv
import glob
import shutil
import string

def find_true_image(true_image_folder, voice_fea_name):
    image_name = voice_fea_name.split('.')[-2]
    Modality, Vocal, Emotion, intensity, Statement, Repetition, Actor = image_name.split('-')
    Actor_folder = "Actor_{}".format(Actor)
    true_image_paths = os.path.join(true_image_folder, Actor_folder, image_name, "*.png")
    true_image_paths = glob.glob(true_image_paths)
    true_image_path = true_image_paths[int(len(true_image_paths) / 2)]
    return true_image_path


def get_voclexb_csv1(voice_data_pth, image_data_pth, csv_pth, data_ext):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    # new_image_folder = "./datasets/RAVDESS/1 image-Actor1-24-single"

    list_name ={"voice":"wav", "image":"png", "mfcc":"mfcc", "fbank":"fbank", "spectrogram":"spectrogram"}
    file_path = list_name[data_ext]
    headers = ['actor_ID','gender','vocal_channel','emotion','emotion_intensity','mfcc_path', 'image_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, dirs, filenames in os.walk(voice_data_pth):      # 音频数据集根目录, 子目录, 文件名
        filenames.sort()
        for filename in filenames:
            if filename.endswith("wav"):              # 校验文件后缀名, wav或者npy
                voice_feat_path = os.path.join(root, filename)
                flag = filename[:-4].split('-')
                if flag[0] == '01':
                    true_image_path = find_true_image(image_data_pth, filename)
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_ID':flag[6], 'gender':gend,'vocal_channel':flag[1],'emotion':flag[2],
                                      'emotion_intensity':flag[3], 'mfcc_path': voice_feat_path, 'image_path': true_image_path})
                    print("voice_feat_path:{0}, true_image_path:{1},actor:{2}".format(voice_feat_path,true_image_path, flag[6]))

    print("number:{}".format(len(data_list)))
    csv_pth = os.path.join(csv_pth, '{}_image_list.csv'.format(list_name[data_ext]))
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)


def get_labels(voice_list, face_list):
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
    temp1 = []
    temp2 = []
    label_dict = dict(zip(names, range(len(names))))

    for name in names:
        for item in voice_list:   # 为list增加序号label_id
            if name == item['name']:
                temp1.append(item['filepath'])
        for item in face_list:
            if name == item['name']:
                temp2.append(item['filepath'])
        voice_face_list.append({'name': item['name'], 'id': label_dict[item['name']], 'voice_path': temp1, 'voice_path': temp2})
        temp1.clear()
        temp2.clear()

    return voice_face_list


def get_dataset_files(data_dir, data_ext, celeb_ids):
    data_list = []

    # read data directory
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder)   #  default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
                data_list.append({'filepath': filepath, 'name': celeb_name})
    return data_list


def get_voclexb_csv(csv_files, voice_folder, face_folder):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    voice_list = []
    face_list = []
    csv_headers = ['actor_ID','name','gender','nation','wav_path', 'image_path']
    triplet_list = []
    actor_dict = {}

    with open(csv_files) as f:
        lines = f.readlines()[1:]
        for line in lines:
            # print(line)
            actor_ID, name, gender, nation, _ = line.rstrip("\n").split('\t')
            actor_dict[actor_ID] = name

    voice_list = get_dataset_files(voice_folder, 'wav', actor_dict)
    face_list = get_dataset_files(face_folder, 'jpg', actor_dict)
    voice_face_list = get_labels(voice_list, face_list)

    csv_pth = os.path.join(csv_pth, 'image_list.csv')
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)

    return actor_dict, len(actor_dict)



if __name__ == '__main__':
    # get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/fbank'

    csv_files = '/home/fz/2-VF-feature/SVHFNet/dataset/voclexb-VGG_face-datasets/vox1_meta.csv'
    voice_folder = '/home/fz/2-VF-feature/SVHFNet/dataset/voclexb-VGG_face-datasets/2-voice-wav'
    face_folder = '/home/fz/2-VF-feature/SVHFNet/dataset/voclexb-VGG_face-datasets/1-face'
    list, num = get_voclexb_csv(csv_files, voice_folder, face_folder)
    pass


    # voice_data_pth = './datasets/RAVDESS/2 wave-Actor1-24-32k'
    # image_data_pth = './datasets/RAVDESS/1 image-Actor1-24-single'
    # csv_pth = "./datasets/RAVDESS"
    # get_RAVDESS_csv(voice_data_pth, image_data_pth, csv_pth, 'voice')


