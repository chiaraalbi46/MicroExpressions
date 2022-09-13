import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, num_classes, drop_val):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # in_channels = 1
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(8, 12, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(12, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(16, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(24, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(48, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        ## added
        self.conv6 = nn.Conv3d(64, 72, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        ## added

        self.fc6 = nn.Linear(1152, 512)
        # self.fc6 = nn.Linear(25088, 4096)  # todo: modo per rendere la prima dim dinamico
        self.fc7 = nn.Linear(512, 64)
        # self.fc8 = nn.Linear(4096, 8)  # le nostre labels sono 7 (le emozioni) + 1 (nessuna)
        self.fc8 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=drop_val)  # 0.2 originariamente

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        #print("h relu su conv1 shape: ", h.shape)
        h = self.pool1(h)
        #print("h pool1 shape: ", h.shape)

        h = self.relu(self.conv2(h))
        #print("h relu su conv2 shape: ", h.shape)
        h = self.pool2(h)
        #print("h pool2 shape: ", h.shape)

        h = self.relu(self.conv3a(h))
        #print("h relu su conv3a shape: ", h.shape)
        h = self.relu(self.conv3b(h))
        #print("h relu su conv3b shape: ", h.shape)
        h = self.pool3(h)
        #print("h pool3 shape: ", h.shape)

        h = self.relu(self.conv4a(h))
        #print("h relu su conv4a shape: ", h.shape)
        h = self.relu(self.conv4b(h))
        #print("h relu su conv4b shape: ", h.shape)
        h = self.pool4(h)
        #print("h relu su pool4 shape: ", h.shape)

        h = self.relu(self.conv5a(h))
        #print("h relu su conv5a shape: ", h.shape)
        h = self.relu(self.conv5b(h))
        #print("h relu su conv5b shape: ", h.shape)
        h = self.pool5(h)
        #print("h relu su pool5 shape: ", h.shape)

        ## added
        h = self.relu(self.conv6(h))
        #print("h relu su conv6 shape: ", h.shape)
        h = self.pool6(h)
        #print("h relu su pool6 shape: ", h.shape)
        ##

        out_fc = h.size(1) * h.size(2) * h.size(3) * h.size(4)
        #print(out_fc)
        h = h.view(-1, out_fc)
        #print("h view shape: ", h.shape)
        h = self.relu(self.fc6(h))
        #print("h view shape: ", h.shape)
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        #print("h view shape: ", h.shape)
        h = self.dropout(h)

        logits = self.fc8(h)
        #print(logits)
        # probs = self.softmax(logits)  # lo fa la cross entropy

        return logits  # probs


"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

if __name__ == '__main__':
    import pickle
    from torch.autograd import Variable
    import torch
    import numpy as np

    # f = open('test_net_200x200.pckl', 'rb')
    # tens = pickle.load(f)
    f = open('video_labels.pckl', 'rb')
    tens, labels = pickle.load(f)
    f.close()
    net = C3D(num_classes=8, drop_val=0.5)
    # # net.cuda()
    # net.eval()
    #
    # # perform prediction
    # # clip = Variable(torch.from_numpy(tens[:, :, 0:16, :, :]))  # 200, 200
    # clip = Variable(torch.from_numpy(tens))  # 200, 200
    # # va bene anche per 50 frame !!
    # # clip = clip.cuda()
    # prediction = net(clip)
    # # loss = torch.nn.CrossEntropyLoss()
    # # # loss_val = loss(prediction, torch.from_numpy(labels))
    # # labels_num = np.float32(np.array((0, 1, 0, 2, 1)))  # 0 felicità, 1 paura, 2 sorpresa
    # # t = torch.from_numpy(labels_num).type(torch.long)  # torch.long serve
    # # loss_val = loss(prediction, t)
    #
    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())  # [x for x in net.parameters()]
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print("Number of parameters: ", params)