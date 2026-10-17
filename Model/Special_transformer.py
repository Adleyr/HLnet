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




        K_unfold = F.normalize(K_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        Q_unfold= F.normalize(Q_unfold, dim=1)  # [N, C*k*k, H*W]


        attention_scores = torch.bmm(Q_unfold, K_unfold)


        attention_scores = attention_scores / torch.sqrt(torch.tensor(K_unfold.size(1)))

        attention_weights = F.softmax(attention_scores, dim=-1)

        Aggregation = torch.bmm(attention_weights, V_unfold)


        R_lv3 = torch.bmm(K_unfold , Q_unfold)  # [N, Hr*Wr, H*W]



        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        ### transfer

        T_lv3_unfold = self.bis(V_unfold, 2, R_lv3_star_arg)


        Tor=torch.softmax(V_unfold * T_lv3_unfold,dim=-1)

        Bridge=F.normalize(Tor + Aggregation,dim=2)



        S = R_lv3_star.view(R_lv3_star.size(0), 1, Q.size(2), Q.size(3))

        SS=self.Conv(S)
        SS_fold=F.unfold(SS, kernel_size=(3, 3), padding=1)
        TRV=SS_fold * Bridge +Aggregation

        FN=F.fold(TRV,output_size=Q.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)

        return FN

