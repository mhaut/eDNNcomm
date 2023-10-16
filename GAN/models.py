import torch
import torch.nn as nn

# build model
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_size, channels, k):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.init_size = img_size // 4
        self.k = k
        self.l1 = nn.Sequential(nn.Linear(latent_dim, (128*self.k) * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128*self.k),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128*self.k, 128*self.k, 3, stride=1, padding=1),
            nn.BatchNorm2d(128*self.k, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128*self.k, 64*self.k, 3, stride=1, padding=1),
            nn.BatchNorm2d(64*self.k, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64*self.k, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128*self.k, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img




class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size, channels, k):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16*k, bn=False),
            *discriminator_block(16*k, 32*k),
            *discriminator_block(32*k, 64*k),
            *discriminator_block(64*k, 128*k),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear((128*k) * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear((128*k) * ds_size ** 2, n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
