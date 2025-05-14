import torch.optim as optim
from util import *
from models import RES
import math
import itertools

def depend_on_mean_feature_sample(category=None,feature=None,cov=None):
    sample=[]
    for i in category:
        mean = feature[i]  # 每个类的均值
        multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples_per_class = multivariate_normal.sample((1,))
        sample.append(samples_per_class)
    sample=torch.cat(sample,dim=0)
    return sample

def ConCP(lab=None,category=None,beta_w=None,lamda_w=None,alpha_w=None,file_name=None,cls=None):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_wipe_out = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    transform_after = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),  # cifar10
    ])
    # 加载所需模型
    path_model='backdoor_model_path'
    para = torch.load(path_model)
    model = RES.ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(para)
    #加载辅助模型
    copy_model = RES.ResNet18()
    copy_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    copy_model.fc = torch.nn.Linear(512, 10)  # 将最后的全连接层改掉
    copy_model.load_state_dict(para)
    model = model.to(device)
    copy_model=copy_model.to(device)
    # 加载祛毒数据
    ori_wipe_out = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,transform=transform_wipe_out)
    ori_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    # 准备训练数据
    attack_trigger = torch.load('/attack_trigger/cifar_' + str(lab) + '.pth')
    clean_data = []  # 存储原始干净的5000张
    patch_data = []  # 存储每个类贴上defense_trigger的数据
    test_data = []  # 存储测试用的数据
    randon_label=cls
    for z in range(10):
        per_data = [[image, label] for image, label in ori_test if label == z]
        wipe_out_data = [[image, label] for image, label in ori_wipe_out if label == z]
        num_wipe = int(len(per_data) * 0.5)
        num_test=int(len(per_data) * 0.5)
        clean_data.extend(wipe_out_data[:num_wipe])
        test_data.extend((per_data[num_test:]))
    for i in randon_label:
        defense_trigger = torch.load('defense_trigger_path')
        patch_trigger = [[torch.clamp(image+defense_trigger[0],0,1), label] for image, label in clean_data]
        patch_data.extend(patch_trigger)
    Loss1_data = MyCIFAR10Dataset(patch_data, transform_after)
    patch_data_dataloader = DataLoader(Loss1_data, batch_size=200, shuffle=True)
    Loss2_data = MyCIFAR10Dataset(clean_data, transform_after)
    clean_data_dataloader = DataLoader(Loss2_data, batch_size=40, shuffle=True)
    loader2_iter = itertools.cycle(clean_data_dataloader)
    # 准备测试ACC数据
    acc_data = MyCIFAR10Dataset(test_data, transform_after)
    acc_data_dataloader = DataLoader(acc_data, batch_size=256, shuffle=True)
    # 准备测试ASR数据
    asr_data = [[image + attack_trigger[0] * 2, label] for image, label in test_data if label != lab]
    asr_data = MyCIFAR10Dataset(asr_data, transform_after)
    asr_data_dataloader = DataLoader(asr_data, batch_size=128, shuffle=True)
    mse = nn.MSELoss()
    criterion = nn.CrossEntropyLoss().to(device)

    param_list = []
    for name, param in model.named_parameters():
        if 'linear' in name or 'fc' in name:
            # pass
            print("要初始化")
            if 'weight' in name:
                print("选择1")
                logging.info(f'Initialize linear classifier weight {name}.')
                std = 1 / math.sqrt(param.size(-1))
                param.data.uniform_(-std, std)
            else:
                print("选择2")
                logging.info(f'Initialize linear classifier weight {name}.')
                param.data.uniform_(-std, std)
        else:
            print("模式2")
            param.requires_grad = True
            param_list.append(param)

    optimizer = optim.SGD(param_list, lr=0.01, momentum=0.9, weight_decay=5e-4)
    sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    feature = torch.tensor(torch.load('./new_data/all_attack_mean_feature/' + str(file_name) + '.pth'))
    cov = torch.load('./new_data/all_attack_cov_feature/' + str(file_name) + '.pth').clone().detach()
    cov += torch.eye(cov.size(0)) * 1e-4
    feature = feature.to(device)
    cov = cov.to(device)

    model.eval()
    total = 0
    correct = 0
    for image, label in acc_data_dataloader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model.forward(image, return_features=False)
        # 得到预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 更新计数器
        total += label.size(0)
        correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print("原始ACC是：" + str(accuracy))

    total = 0
    correct = 0
    for image, label in asr_data_dataloader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model.forward(image, return_features=False)
        # 得到预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 更新计数器
        total += label.size(0)
        correct += (predicted == lab).sum().item()
    asr = 100 * correct / total
    print("原始ASR是：" + str(asr))

    for epoch in range(1):
        loss_list = []
        model.train()
        dataloader1_iter = iter(patch_data_dataloader)
        dataloader2_iter = iter(clean_data_dataloader)
        num_batches = max(len(dataloader1_iter), len(dataloader2_iter))
        for batch_idx in range(num_batches):
            # 从第一个 dataloader 取数据
            inputs1, targets1 = next(dataloader1_iter)
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            # 从第二个 dataloader 取数据
            inputs2, targets2 = next(loader2_iter)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            # 前向传播，分别计算两个损失
            outputs1 = model.forward(inputs1, return_features=False)
            outputs3=model.forward(inputs2, return_features=True)
            outputs4=copy_model.forward(inputs2, return_features=True)
            outputs2=depend_on_mean_feature_sample(targets1,feature,cov)
            outputs2=outputs2.to(device)
            outputs5 = model.forward(inputs1, return_features=True)
            loss1=mse(outputs2, outputs5)
            loss2=mse(outputs3,outputs4)
            loss3 = criterion(outputs1, targets1)
            loss =beta_w*loss1+lamda_w*loss2+alpha_w*loss3
            optimizer.zero_grad()
            loss.backward()
            loss_list.append(float(loss.data))
            optimizer.step()
        sche.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %.06f' % (epoch, ave_loss))

        # if epoch % 3 == 0 or epoch == 29:
        model.eval()
        total = 0
        correct = 0
        for image, label in acc_data_dataloader:
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                outputs = model.forward(image, return_features=False)
            # 得到预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 更新计数器
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = 100 * correct / total
        print("ACC是：" + str(accuracy))

        # if epoch % 3 == 0 or epoch == 29:
        total = 0
        correct = 0
        for image, label in asr_data_dataloader:
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                outputs = model.forward(image, return_features=False)
            # 得到预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 更新计数器
            total += label.size(0)
            correct += (predicted == lab).sum().item()
        asr = 100 * correct / total
        print("ASR是：" + str(asr))

