import torch
import torch.nn as nn
import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self,in_c):
        super(Transformer, self).__init__()

        self.Conv=nn.Sequential(*[nn.Conv2d(in_channels=1,out_channels=in_c,kernel_size=1),
                                  nn.Conv2d(in_channels=in_c,out_channels=in_c,kernel_size=3,padding=1)])
    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, V, K, Q):

        ### search
        Q_unfold=F.unfold(Q, kernel_size=(3, 3), padding=1)
        K_unfold=F.unfold(K,kernel_size=(3,3),padding=1)
        V_unfold = F.unfold(V, kernel_size=(3, 3), padding=1)
        K_unfold = K_unfold.permute(0, 2, 1)
        # print("K_unfold:",K_unfold.shape)   ##K_unfold: torch.Size([1, 1024, 72])
        # print("Q_unfold:",Q_unfold.shape)   ##Q_unfold: torch.Size([1, 72, 1024])
        # print("V_unfold:",V_unfold.shape)   ##V_unfold: torch.Size([1, 72, 1024])



        K_unfold = F.normalize(K_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        Q_unfold= F.normalize(Q_unfold, dim=1)  # [N, C*k*k, H*W]
        # print("K_unfold:",K_unfold.shape)   ##K_unfold: torch.Size([1, 1024, 72])
        # print("Q_unfold:",Q_unfold.shape)   ##Q_unfold: torch.Size([1, 72, 1024])
        # print("V_unfold:",V_unfold.shape)   ##V_unfold: torch.Size([1, 72, 1024])

        # 1. 计算Q和K的点积
        attention_scores = torch.bmm(Q_unfold, K_unfold)

        # 2. 除以K的维度的平方根
        attention_scores = attention_scores / torch.sqrt(torch.tensor(K_unfold.size(1)))
        # print("attention_scores",attention_scores.shape)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # print("attention_weights:",attention_weights.shape)
        Aggregation = torch.bmm(attention_weights, V_unfold)
        # print("output:",Aggregation.shape)

        R_lv3 = torch.bmm(K_unfold , Q_unfold)  # [N, Hr*Wr, H*W]



        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        ### transfer

        T_lv3_unfold = self.bis(V_unfold, 2, R_lv3_star_arg)
        # print("T_lv3_unfold.shape:",T_lv3_unfold.shape)

        Tor=torch.softmax(V_unfold * T_lv3_unfold,dim=-1)

        Bridge=F.normalize(Tor + Aggregation,dim=2)


        # T_lv3 = F.fold(T_lv3_unfold, output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, Q.size(2), Q.size(3))

        SS=self.Conv(S)
        SS_fold=F.unfold(SS, kernel_size=(3, 3), padding=1)
        TRV=SS_fold * Bridge +Aggregation

        FN=F.fold(TRV,output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        return FN


# net=Transformer(64).cuda()
#
# V = torch.randn(1, 64, 64, 64).cuda()  # 输入数据，通常是图像
# K = torch.randn(1, 64, 64, 64).cuda()  # 输入数据，通常是图像
# Q = torch.randn(1, 64, 64, 64).cuda()  # 输入数据，通常是图像
#
# out=net(V,K,Q)
#
# print(out.shape)