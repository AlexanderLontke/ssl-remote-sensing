
# import torch

# class VAELoss():

#     def __init__(self):
#         super.__init__
    
#     def gaussian_likelihood(x_hat, logscale, x):
#         scale = torch.exp(logscale)
#         mean = x_hat
#         dist = torch.distributions.Normal(mean, scale)

#         # measure prob of seeing image under p(x|z)
#         log_pxz = dist.log_prob(x)
#         return log_pxz.sum(dim=(1, 2, 3))
    
#     def kl_divergence(self, z, mu, std):
#         # --------------------------
#         # Monte carlo KL divergence
#         # --------------------------
#         # 1. define the first two probabilities (in this case Normal for both)
#         p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
#         q = torch.distributions.Normal(mu, std)

#         # 2. get the probabilities from the equation
#         log_qzx = q.log_prob(z)
#         log_pz = p.log_prob(z)

#          # kl
#         kl = (log_qzx - log_pz)
#         kl = kl.sum(-1)
#         return kl
    


    