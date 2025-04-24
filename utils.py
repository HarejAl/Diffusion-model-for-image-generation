import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def Load_images(N):
    import tensorflow as tf
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    digits = (np.float32(x_train)/255).reshape(60000,1,28,28)[:N]
    labels = np.float32(y_train)[:N].reshape(-1,1)
    
    return digits, labels


# Sinusoidal positional embedding for the time variable.
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        # t is assumed to be of shape [batch_size]
        device = t.device
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)   #[batch_size, 1, embedding_dim]
        return emb.squeeze(1)

## Needs to embedd both the time step and the condition (digit value)
class TwoBranchNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, activation_fn):
        """
        Neural net having two input branches for encodding the:
            - time step-> FCNN
            - map of the holes -> CNN
        Args:
            hidden_dim: Number of hidden units in each branch.
            output_dim: Size of the final output vector (N) -> n_chanels.
        """
        super().__init__()
        
        
        self.hidden_dim = hidden_dim
        
        self.branch1= SinusoidalTimeEmbeddings(hidden_dim)
        
        self.branch2 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
    
        # Merging network: Combines outputs from the two branches.
        self.merged = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            activation_fn,
        )
        
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor of shape [batch, 1] -> time_step
            x2: Tensor of shape [batch, 1] -> condition
        Returns:
            Tensor of shape [batch, output_dim]
        """
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        # Concatenate along the feature dimension.
        merged = torch.cat([out1, out2], dim=1)  # [batch, hidden_dim*2]
        # Process the merged vector.
        output = self.merged(merged)  # [batch, output_dim] #output_dim = channels
        return output

# A simple convolutional block that performs a 3x3 convolution and adds a time- and condition-conditioned bias.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        # self.double_conv = double_conv(in_channels, out_channels)
        self.conv_layer1 = nn.Conv2d(in_channels,out_channels, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(out_channels,out_channels, 3, padding=1)
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.act_fcn = activation_fn
        self.emb_projection = TwoBranchNet(64, out_channels, activation_fn)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_step, condition):
        h = self.conv_layer1(x)
        emb_proj = self.emb_projection(time_step,condition).unsqueeze(-1).unsqueeze(-1) # unsqueeze needs to transform it the correct duimension
        h = h + emb_proj
        h = self.act_fcn(h)
        h = self.conv_layer2(h)
        h = self.act_fcn(h) + self.res_conv(x) # This is very important
        
        return h

# A simple convolutional network that embeds both time and a continuous condition.
class UNet(nn.Module):
    def __init__(self, in_chan=1, out_channels=1, base_ch = 64, activation_fn = nn.SiLU()):
        """
        U net without skip connections 
        Args:
            in_channels: Number of channels in the input (1 for grayscale, 3 for rgb)
            in_channels: Number of channels in the output (1 for grayscale, 3 for rgb)
        """
        super().__init__()
        
        self.input_conv = nn.Conv2d(1, base_ch, kernel_size=3, padding=1)
        self.down1 = ConvBlock(in_channels=base_ch, out_channels=2*base_ch, activation_fn = activation_fn)
        self.down2 = ConvBlock(in_channels=2*base_ch, out_channels=4*base_ch, activation_fn = activation_fn)
        self.down3 = ConvBlock(in_channels=4*base_ch, out_channels=8*base_ch, activation_fn = activation_fn)
    
        self.mid = ConvBlock(in_channels=8*base_ch, out_channels=8*base_ch, activation_fn = activation_fn)
        
        self.up3 = ConvBlock(in_channels=16*base_ch, out_channels=4*base_ch, activation_fn = activation_fn)
        self.up2 = ConvBlock(in_channels=8*base_ch, out_channels=2*base_ch, activation_fn = activation_fn)
        self.up1 = ConvBlock(in_channels=4*base_ch, out_channels= base_ch, activation_fn = activation_fn)
        
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.downsampling = nn.MaxPool2d(kernel_size=2)
        
        self.output_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x, t, condition):
        """
        Args:
            x: Input noise tensor of shape [batch, in_channels, height, width].
            t: Tensor of time steps of shape [batch].
            condition: Tensor of continuous condition values with shape [batch, condition_dim].
        """

        x0 = self.input_conv(x)
        
        x1_skip= self.down1(x0, t, condition)
        x1 = self.downsampling(x1_skip) # This is very important: downsampling after saving the skip connection
        
        x2_skip = self.down2(x1, t, condition)
        x2 = self.downsampling(x2_skip)
        
        x3 = self.down3(x2, t, condition)
        
        x_mid= self.mid(x3, t, condition)
        
        x_mid = torch.cat((x_mid,x3),axis=1)
        
        x2 = self.up3(x_mid, t, condition)
        
        x2 = self.upsampling(x2)
        
        if x2.shape[2:] != x2_skip.shape[2:]: ## Inteprolation is needed in case the two concatenating arrays do not share the same shape
            x2 = F.interpolate(x2, size=x2_skip.shape[2:], mode="bilinear", align_corners=False)
        
        x2 = torch.cat((x2,x2_skip),axis=1)
        
        x1 = self.up2(x2, t, condition) 
        
        x1 = self.upsampling(x1)
        
        if x1.shape[2:] != x1_skip.shape[2:]: 
            x1 = F.interpolate(x1, size=x1_skip.shape[2:], mode="bilinear", align_corners=False)
        
        x1 = torch.cat((x1,x1_skip),axis=1)
        
        x0 = self.up1(x1, t, condition)
        
        x_out = self.output_conv(x0)
        
        return x_out

class Diffusion():
    
    def __init__(self, T = 100, beta_min = 1e-4 , beta_max = 0.2, device = torch.device('cpu')):
        """
        Parameters
        ----------
        T : int, number of time steps. The default is 100.
        beta_min : fl, min variance of noise. The default is 1e-4.
        beta_max : fl, max variance of noise. The default is 0.2.
        device : torch.device(). The default is torch.device('cpu').

        """
        
        self.T = T
        self.beta = torch.linspace(beta_min, beta_max, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)  # Compute cumulative product \bar{\alpha}_t
        self.model = UNet().to(device)
        self.device = device

    def diffusion_loss(self, model, x0, t, condition, batch_size):
        """Computes the score-matching loss for diffusion training"""
        noise = torch.randn_like(x0).to(self.device)
        alpha_bar_t = self.alpha_bar[t.long()].view(batch_size, 1, 1, 1) #Transform the variable in the opportune shape
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        noise_pred = model(xt, t, condition)
        loss = torch.mean((noise - noise_pred) ** 2)
        return loss
    
    def training(self, num_epochs, dataloader, lr = 1e-4):
        """
        Trains the UNet model to generate new images

        Parameters
        ----------
        num_epochs : int, number of training epochs
        dataloader : dataloader, dataloader organized in: (image, label)
        lr : fl, learning rate. The default is 1e-4.
        """
        
        optimizer = optim.Adam(self.model.parameters(), lr)
        best_loss = 1e3; loss_tracker = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_x, batch_y in dataloader:  # Iterate over 5 batches
                batch_size = len(batch_x[:,0,0,0])
                for _ in range(25): # 25 iterations on the same batch: it shows very good results
                    
                    # Samplea random time sample to the training
                    t = torch.randint(0, self.T, (batch_size,1),dtype=torch.float32).to(self.device) # .long() converts int in int64
                    # Compute loss
                    loss = self.diffusion_loss(self.model, batch_x, t, batch_y, batch_size)
                
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()/batch_size
                
            loss_tracker.append(epoch_loss)
        
            if epoch_loss < best_loss:
                torch.save(self.model,'Diffusion_model.pth')
                best_loss = epoch_loss
                
            print('Epoch: {}, Loss {:.4e}'.format(epoch+1, epoch_loss))
            
        # Plotting the training loss 
        fig, ax1 = plt.subplots(1,1,figsize=(6,4))
        ax1.set_yscale('log')
        ax1.plot(loss_tracker,'b',label='Training loss')
        ax1.set_title('TRAINING LOSS')
        ax1.legend()
        ax1.grid(visible=True)
                    
    def sample(self, batch_size, image_size = 28):
        """
        Generates an image by running the reverse diffusion process.
        
        Parameters
        ----------
        image_size : int, number of pixels on each axis (original MNIST images are 28 x 28). It is possible to change but the model was trained just on 28x28 images. The default is 28.
        batch_size : number of sampled images
        """
        device = self.device
        
        # Start from pure noise
        x_t = torch.randn((batch_size, 1, image_size, image_size)).to(device)  
        # Sample random labels for generating digits 
        y = torch.randint(0, 10, (batch_size,1),dtype=torch.float32).to(device)
        frames = []; frames_t = []
    
        for t in reversed(range(self.T)):  # Reverse diffusion loop
        
            t_tensor = torch.tensor(np.ones((batch_size,1))*t,dtype=torch.float32).to(device)
            
            if t == 0:
                noise = torch.zeros_like(x_t).to(device)
            else:
                noise = torch.randn_like(x_t).to(device)
    
            # Predict the noise
            noise_pred = self.model(x_t, t_tensor, y)
    
            # Compute x_{t-1} using the reverse equation
            alpha_t = self.alpha[t].view(1, 1, 1, 1).to(device)
            beta_t = self.beta[t].view(1, 1, 1, 1).to(device)
            alpha_bar_t = self.alpha_bar[t].view(1, 1, 1, 1) #Transform the variable in teh opportune shape
            sigma_t = torch.sqrt(beta_t)
            
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t) / torch.sqrt(1-alpha_bar_t) * noise_pred) + sigma_t * noise
            
            if t % 10 == 0 or t == self.T-1:
                frames.append(x_t.cpu().detach().numpy()[1,0,:,:])
                frames_t.append(t)
    
        return x_t.cpu().detach().numpy()[:,0,:,:], y.cpu().detach().numpy(), frames, frames_t # Return the final generated image


