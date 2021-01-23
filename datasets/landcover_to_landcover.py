from torch.utils.data import DataLoader
import imageio
from torch.utils.data import Dataset
from os.path import join
import numpy as np
import glob
import json
from os.path import basename
import matplotlib.cm as cm
from torch.nn import Softmax2d
import torch
import matplotlib.pyplot as plt
from utils.image_type import OSO,CLC
from torchvision.transforms import Compose
from datasets.transforms import ToTensor,ToOneHot

class LandcoverToLandcover(Dataset):
    def __init__(self,  root_dir,src_year,tgt_year,sea_value=15,mode="train",ext=".npy",transform=None):
        self.root_dir = root_dir
        self.mode=mode

        src = "oso_" + str(src_year)
        tgt="clc_" + str(tgt_year)

        self.current_dir=join(self.root_dir,mode)

        with open(join(root_dir,"metadata.json")) as f:
          self.metadata=json.load((f))

        
        self.d=128
        self.d=int(self.d/2)
        self.d_i=np.arange(0,self.d/2)
        self.freq=1/(10000**(2*self.d_i/self.d))

        self.list_image_src=glob.glob(join(self.current_dir,src+"/*"+ext))
        self.list_image_tgt=[i.replace(src,tgt) for i in self.list_image_src]
        self.transform=transform
        self.sea_value=sea_value

    def __len__(self):
        return len (self.list_image_src)

    def __getitem__(self, idx):
        with torch.no_grad():
            image_src = torch.from_numpy(np.load(self.list_image_src[idx])).float()
      
            # image_src=torch.where((image_src==5.0)|(image_src==6.0)|(image_src==7.0),torch.tensor([12.0]),image_src).float()
            image_tgt = torch.from_numpy(np.load(self.list_image_tgt[idx])).float()

            sea = torch.where(image_tgt==self.sea_value,torch.tensor([1]),torch.tensor([0])).float()
            metadata=self.metadata["oso_2018"][basename(self.list_image_src[idx]).replace(".npy",".tif")]["gt"]
            x,y=metadata[0],metadata[3]
            x,y=x/10000,y/10000
            enc=np.zeros(self.d*2)
            enc[0:self.d:2]=np.sin(x * self.freq)
            enc[1:self.d:2]=np.cos(x * self.freq)
            enc[self.d::2]=np.sin(y * self.freq)
            enc[self.d+1::2]=np.cos(y * self.freq)
            sample = [[image_src,sea,torch.tensor(enc).float(),self.list_image_src[idx]],[image_tgt,]]
        if self.transform:
            sample = self.transform(sample)
        return sample

class LandcoverToLandcoverDataLoader:
    def __init__(self, config,mode="normal"):
        """
        :param config:
        """
        self.config = config
        self.train_src_info,self.train_tgt_info,self.test_src_info,self.test_tgt_info = self.get_srctgt_type()
        if mode == "normal":
            self.train_loader = DataLoader(LandcoverToLandcover(config.data_folder,self.train_src_info.year,self.train_tgt_info.year,sea_value=config.sea_value,mode="train",ext=".npy",transform=Compose([ToOneHot(len(self.train_tgt_info.id_labels))])), batch_size=self.config.train_batch_size, shuffle=True)
            self.valid_loader = DataLoader(LandcoverToLandcover(config.data_folder,self.train_src_info.year,self.train_tgt_info.year,sea_value=config.sea_value,mode="val",ext=".npy",transform=Compose([ToOneHot(len(self.train_tgt_info.id_labels))])), batch_size=self.config.valid_batch_size, shuffle=False)
            self.test_loader = DataLoader(LandcoverToLandcover(config.data_folder,self.test_src_info.year,self.test_tgt_info.year,sea_value=config.sea_value,mode="test",ext=".npy",transform=Compose([ToOneHot(len(self.test_tgt_info.id_labels))])), batch_size=self.config.test_batch_size, shuffle=False)

            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
            self.test_iterations= len(self.test_loader)
        elif mode=="full":
            self.full_loader = DataLoader(LandcoverToLandcover(config.data_folder,self.train_src_info.year,self.train_tgt_info.year,sea_value=config.sea_value,mode="full",ext=".npy",transform=Compose([ToOneHot(len(self.train_tgt_info.id_labels))])), batch_size=self.config.train_batch_size, shuffle=True)
            self.full_iterations = len(self.full_loader)
        else:
            raise Exception("Unknow dataset mode")


    def get_srctgt_type(self):
        train_src_info = self.config.train_src_info
        train_tgt_info = self.config.train_tgt_info
        test_src_info = self.config.test_src_info
        test_tgt_info = self.config.test_tgt_info

        def criterion(info):
            if info["name"].upper() == "OSO":
                return OSO(year=info["year"])
            elif info["name"].upper() == "CLC":
                return CLC(level=info["level"], year=info["year"])
            else:
                raise ValueError("Unknow image type : " + info["name"])

        return criterion(train_src_info), criterion(train_tgt_info),criterion(test_src_info), criterion(test_tgt_info)

    def plot_samples_per_epoch(self, inputs,targets,outputs, epoch):
        """
        Plotting the batch images
        :param inputs: Tensor source Land-cover of shape (B,1,H,W)
        :param targets: Tensor one-hot-encoded targets Land-cover of shape (B,n,H,W) with n (number of labels)
        :param outputs: Tensor one-hot-encoded prediction Land-cover of shape (B,n,H,W) with n (number of labels)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """

        def masks_to_img( masks):
            return np.argmax(masks, axis=0) + 1

        def probability_map( output):
            m = Softmax2d()
            prob, _ = torch.max(m(output), dim=1)
            return prob

        def colorbar( fig, ax, cmap, labels_name):
            n_labels = len(labels_name)
            mappable = cm.ScalarMappable(cmap=cmap)
            mappable.set_array([])
            mappable.set_clim(0.5, n_labels + 0.5)
            colorbar = fig.colorbar(mappable, ax=ax)
            colorbar.set_ticks(np.linspace(1, n_labels + 1, n_labels + 1))
            colorbar.set_ticklabels(labels_name)
            if len(labels_name) > 30:
                colorbar.ax.tick_params(labelsize=5)
            else:
                colorbar.ax.tick_params(labelsize=15)

        def normalize_value_for_diplay( inputs, src_type):
            for i in range(len(src_type.id_labels)):
                inputs = np.where(inputs == src_type.id_labels[i], i + 1, inputs)
            return inputs

        def show_res(inputs,src_type, labels, tgt_type, outputs, save_path, display=False):
            fig, axs = plt.subplots(3, 3, figsize=(30, 20))
            for i in range(3):
                show_labels = masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
                if outputs.shape[1] > 1:

                    show_outputs = masks_to_img(outputs.cpu().detach().numpy()[i]).astype("uint8")
                else:
                    show_outputs = outputs.cpu().detach().numpy()[i][0]
                input = inputs.cpu().detach().numpy()[i][0]
                input = normalize_value_for_diplay(input, src_type)

                axs[i][0].imshow(input, cmap=src_type.matplotlib_cmap, vmin=1, vmax=len(src_type.labels_name),
                                 interpolation='nearest')
                axs[i][0].axis('off')
                colorbar(fig, axs[i][0], src_type.matplotlib_cmap, src_type.labels_name)

                axs[i][1].imshow(show_labels, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name),
                                 interpolation='nearest')
                axs[i][1].axis('off')
                colorbar(fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

                axs[i][2].imshow(show_outputs, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name),
                                 interpolation='nearest')
                axs[i][2].axis('off')
                colorbar(fig, axs[i][2], tgt_type.matplotlib_cmap, tgt_type.labels_name)

            # plt.tight_layout()
            fig.savefig(save_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close(fig)
            fig = None

        def show_probability_map( inputs, src_type, labels, tgt_type, outputs, save_path, display=False):
            fig, axs = plt.subplots(3, 3, figsize=(30, 20))
            for i in range(3):
                show_labels = masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
                show_outputs = masks_to_img(outputs.cpu().detach().numpy()[i]).astype("uint8")
                prob = probability_map(outputs).cpu().detach().numpy()[i]

                p = axs[i][2].imshow(prob, cmap='magma', vmin=0, vmax=1)
                axs[i][2].axis('off')
                fig.colorbar(p, ax=axs[i][2])

                axs[i][0].imshow(show_labels, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name),
                                 interpolation='nearest')
                axs[i][0].axis('off')
                colorbar(fig, axs[i][0], tgt_type.matplotlib_cmap, tgt_type.labels_name)

                axs[i][1].imshow(show_outputs, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name),
                                 interpolation='nearest')
                axs[i][1].axis('off')
                colorbar(fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

            # plt.tight_layout()
            fig.savefig(save_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close(fig)
            fig = None

        img_epoch = '{}samples_res_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        show_res(inputs[0: 3], self.test_src_info, targets[0: 3], self.test_tgt_info, outputs[0: 3], img_epoch)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)


    def finalize(self):
        pass
