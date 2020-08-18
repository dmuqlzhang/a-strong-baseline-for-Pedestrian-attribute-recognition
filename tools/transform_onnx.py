import torch.onnx
import torch
from torch.autograd import Variable
from models.osnet import osnet_ain_x1_0
from models.base_block import FeatClassifier, BaseClassifier, BaseClassifier_osnet
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pickle

if __name__=='__main__':
    trans_flag = False
    if trans_flag:
        model_path = '../exp_result/VSP/VSP/img_model/ckpt_max.pth'
        saved_state_dict = torch.load(model_path)['state_dicts']
        saved_state_dict = {k.replace('module.', ''): v for k, v in saved_state_dict.items()}
        # saved_state_dict = {k.replace('backbone.conv2.0.conv1.', 'module.backbone.conv2.0.conv1.'): v for k, v in saved_state_dict.items()}
        backbone = osnet_ain_x1_0(num_classes=56, pretrained=True, loss='softmax')
        classifier = BaseClassifier_osnet(nattr=56)
        torch_model = FeatClassifier(backbone, classifier)
        torch_model.load_state_dict(saved_state_dict)
        input = torch.ones(1, 3, 256, 128)
        # torch2onnx
        # torch_out = torch.onnx.export(torch_model, input, "person_Atrr_osnet_tmp.onnx", export_params=True, verbose=True)
    else:# inf
        # inference eg
        # att_list
        dataset_info = pickle.load(open('../dataset/preprocess/data/VSP/dataset.pkl', 'rb+'))
        att_list = dataset_info.attr_name
        # print(att_list)

        #
        import onnxruntime
        session = onnxruntime.InferenceSession("person_Atrr_osnet.onnx")
        print("The model expects input shape: ", session.get_inputs()[0].shape)
        IN_IMAGE_H = session.get_inputs()[0].shape[2]
        IN_IMAGE_W = session.get_inputs()[0].shape[3]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        thansform_inf = T.Compose([
            T.Resize((IN_IMAGE_H, IN_IMAGE_W)),
            T.ToTensor(),
            normalize
        ])
        img = '../imgs/01009.jpg'
        img_array = Image.open(img).convert('RGB')
        img_array = thansform_inf(img_array)
        img_array = np.array(img_array).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array.to('cuda')
        input_name = session.get_inputs()[0].name
        import datetime
        for i in range(1):
            s = datetime.datetime.now()
            outputs = session.run(None, {input_name: img_array})
            e = datetime.datetime.now()
            print(e-s)
        att_id = list(np.where(np.array(outputs[0][0]) > 0)[0])
        print(type(att_id[0]))
        for id, att in enumerate(att_list):
            if id in att_id:
                print(att)




