# import torch
# from pretext_tasks.vae.loss import VAELoss

# class VAETraining():

#     def __init__(self,model = None):
#         super().__init__()
#         self.model = model

#     def training_step(model, batch):

#         x = batch
#         x = x.float()
#         loss = VAELoss()

#         # encode x to get the mu and variance parameters
#         x_encoded = model.encoder(x)
#         mu, log_var = model.fc_mu(x_encoded), model.fc_var(x_encoded)

#         # sample z from q
#         std = torch.exp(log_var / 2)
#         q = torch.distributions.Normal(mu, std)
#         z = q.rsample()

#         # decoded
#         x_hat = model.decoder(z)

#         # reconstruction loss
#         recon_loss = loss.gaussian_likelihood(x_hat, model.log_scale, x)

#         # kl
#         kl = loss.kl_divergence(z, mu, std)

#         # elbo
#         elbo = (kl - recon_loss)
#         elbo = elbo.mean()

#         return x_encoded,x_hat,elbo

#     def training
