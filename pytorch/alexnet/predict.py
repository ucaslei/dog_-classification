import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(#图片预处理
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
pics=['../1.jpg',"../2.jpg","../3.jpg","../4.jpg",'../5.jpg',"../6.jpg","../7.jpg","../8.jpg","../9.jpg"]

j=1
plt.figure()
for i in pics:

    img = Image.open(i)  #img = Image.open("../tulip.jpg")
    plt.subplot(250+j)
    j=j+1
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)#扩充维度

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create modelpython
    model = AlexNet(num_classes=5)
    # load model weights
    model_weight_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    with torch.no_grad(): #不去跟踪变量损失梯度
    # predict class
        output = torch.squeeze(model(img)) #压缩batch维度
        predict = torch.softmax(output, dim=0) #变成概率分布
        predict_cla = torch.argmax(predict).numpy() #获取概率最大所对应索引值
    #print(class_indict[str(predict_cla)], predict[predict_cla].item())
    plt.title(class_indict[str(predict_cla)]+'   '+str(round(predict[predict_cla].item()*100,2))+'%')
plt.show()
