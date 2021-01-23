import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from torch.nn import Softmax2d
import torch

class plt_loss(object):
    def __call__(self, train_loss, valid_loss, figsize=(10, 10), savefig=None, display=False):
        train_loss=np.array(train_loss)
        valid_loss=np.array(valid_loss)
        f = plt.figure(figsize=figsize)
        plt.plot(train_loss[:,0],train_loss[:,1],c='blue', label="training loss")
        plt.plot(valid_loss[:,0], valid_loss[:,1],c='green', label="validation loss")

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None

class plt_kappa(object):
    def __call__(self, train_kappa, valid_kappa, figsize=(10, 10), savefig=None, display=False):
        train_kappa=np.array(train_kappa)
        valid_kappa=np.array(valid_kappa)
        f = plt.figure(figsize=figsize)
        plt.plot(train_kappa[:,0],train_kappa[:,1],c='blue', label="training kappa")
        plt.plot(valid_kappa[:,0], valid_kappa[:,1],c='green', label="validation kappa")

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None
        
class plt_scatter(object):

    def __call__(self,X,Y,C,label,xlabel,ylabel,figsize=(10,10),savefig=None,display=False):
        f=plt.figure(figsize=figsize)
        for x,y,c,lab in zip(X,Y,C,label):
            plt.scatter(x,y,c=np.array([c]),label=lab,edgecolors='black',s=100)

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if savefig :
            f.savefig(savefig,bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f=None

class plt_image(object):

    def __init__(self):
        pass

    def masks_to_img(self,masks):
        return np.argmax(masks, axis=0)+1

    def probability_map(self,output):
        m = Softmax2d()
        prob, _ = torch.max(m(output), dim=1)
        return prob

    def colorbar(self,fig, ax, cmap, labels_name):
        n_labels = len(labels_name)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(0.5, n_labels + 0.5)
        colorbar = fig.colorbar(mappable, ax=ax)
        colorbar.set_ticks(np.linspace(1, n_labels + 1, n_labels + 1))
        colorbar.set_ticklabels(labels_name)
        if len(labels_name)>30:
            colorbar.ax.tick_params(labelsize=5)
        else:
            colorbar.ax.tick_params(labelsize=15)
            
    def normalize_value_for_diplay(self,inputs,src_type):
        for i in range (len(src_type.id_labels)):
            inputs=np.where(inputs==src_type.id_labels[i],i+1,inputs)
        return inputs

    def show_res(self,inputs, src_type, labels, tgt_type, outputs,save_path,display=False):

        fig, axs = plt.subplots(3, 3, figsize=(30, 20))
        for i in range(3):
            show_labels = self.masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
            if outputs.shape[1]>1:

                show_outputs = self.masks_to_img(outputs.cpu().detach().numpy()[i]).astype("uint8")
            else:
                show_outputs=show_outputs.cpu().detach().numpy()[i][0]
            input = inputs.cpu().detach().numpy()[i][0]
            input =self.normalize_value_for_diplay(input,src_type)

            axs[i][0].imshow(input, cmap=src_type.matplotlib_cmap, vmin=1, vmax=len(src_type.labels_name), interpolation='nearest')
            axs[i][0].axis('off')
            self.colorbar(fig, axs[i][0], src_type.matplotlib_cmap, src_type.labels_name)

            axs[i][1].imshow(show_labels, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
            axs[i][1].axis('off')
            self.colorbar(fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

            axs[i][2].imshow(show_outputs, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
            axs[i][2].axis('off')
            self.colorbar(fig, axs[i][2], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        # plt.tight_layout()
        fig.savefig(save_path,bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig=None

    def show_one_res(self,inputs, src_type, labels, tgt_type, outputs,save_path,display=False):

        fig, axs = plt.subplots(1, 3, figsize=(30, 20))
        show_labels = self.masks_to_img(labels.cpu().numpy()[0]).astype("uint8")
        if outputs.shape[1]>1:

            show_outputs = self.masks_to_img(outputs.cpu().detach().numpy()[0]).astype("uint8")
        else:
            show_outputs=outputs.cpu().detach().numpy()[0][0]
        input = inputs.cpu().detach().numpy()[0][0]
        input =self.normalize_value_for_diplay(input,src_type)

        axs[0].imshow(input, cmap=src_type.matplotlib_cmap, vmin=1, vmax=len(src_type.labels_name), interpolation='nearest')
        axs[0].axis('off')
        self.colorbar(fig, axs[0], src_type.matplotlib_cmap, src_type.labels_name)

        axs[1].imshow(show_labels, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
        axs[1].axis('off')
        self.colorbar(fig, axs[1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        axs[2].imshow(show_outputs, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
        axs[2].axis('off')
        self.colorbar(fig, axs[2], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        # plt.tight_layout()
        fig.savefig(save_path,bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig=None

    def show_probability_map(self,inputs, src_type, labels, tgt_type, outputs,save_path,display=False):
        fig, axs = plt.subplots(3, 3, figsize=(30, 20))
        for i in range(3):
            show_labels = self.masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
            show_outputs = self.masks_to_img(outputs.cpu().detach().numpy()[i]).astype("uint8")
            prob = self.probability_map(outputs).cpu().detach().numpy()[i]

            p = axs[i][2].imshow(prob, cmap='magma', vmin=0, vmax=1)
            axs[i][2].axis('off')
            fig.colorbar(p, ax=axs[i][2])

            axs[i][0].imshow(show_labels, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
            axs[i][0].axis('off')
            self.colorbar(fig, axs[i][0], tgt_type.matplotlib_cmap, tgt_type.labels_name)

            axs[i][1].imshow(show_outputs, cmap=tgt_type.matplotlib_cmap, vmin=1, vmax=len(tgt_type.labels_name), interpolation='nearest')
            axs[i][1].axis('off')
            self.colorbar(fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        # plt.tight_layout()
        fig.savefig(save_path,bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig=None
