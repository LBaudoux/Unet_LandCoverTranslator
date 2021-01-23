"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import shutil
import random
from utils.metrics import EdgePreservationAssessment
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from os.path import basename

from agents.base import BaseAgent

from graphs.models.translating_unet import TranslatingUnet
from graphs.losses.define_loss import define_loss,compute_loss
from datasets.landcover_to_landcover import LandcoverToLandcoverDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

from os.path import join

from utils.plt_utils import plt_scatter,plt_image,plt_loss,plt_kappa

from utils.tensorboardx_utils import tensorboard_summary_writer

import csv

cudnn.benchmark = True


class TranslatingUnetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = TranslatingUnet(self.config)

        # define data_loader
        self.data_loader = LandcoverToLandcoverDataLoader(config=config)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # define loss
        self.loss = define_loss(self.config,self.device)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        if self.config.tensorboard:
            self.summary_writer, self.tensorboard_process = tensorboard_summary_writer(config,comment=self.config.exp_name)


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """

        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

        if self.cuda and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)


    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0,
                        historical_storage="/work/scratch/baudoulu/train2012_valid2018_modelsave/"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        if torch.cuda.device_count() > 1 and self.cuda:
            state = {
                'epoch': self.current_epoch,
                'iteration': self.current_iteration,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'manual_seed': self.manual_seed
            }
        else:
            state = {
                'epoch': self.current_epoch,
                'iteration': self.current_iteration,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'manual_seed': self.manual_seed
            }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name, self.config.checkpoint_dir + 'model_best.pth.tar')


    def run(self):
        """
        The main operator
        :return:
        """
        try:
            torch.cuda.empty_cache()
            self.train()
            # self.save_patch_loss(self.model,self.device,self.data_loader.train_loader)
            torch.cuda.empty_cache()
            self.test()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")


    def train(self):
        """
        Main training loop
        :return:
        """
        loss_ref = 1000
        plot_training_loss = []
        plot_validation_loss = []

        self.logger.info("Start training !")
        for epoch in range(1, self.config.max_epoch + 1):
            self.logger.info("Training epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                        (epoch - 1) / self.config.max_epoch * 100))
            train_loss = self.train_one_epoch()
            plot_training_loss.append([epoch, np.mean(train_loss)])
            torch.cuda.empty_cache()
            if epoch % self.config.validate_every == 0:
                self.logger.info("Validation epoch {}/{} ({:.0f}%):\n".format(epoch, self.config.max_epoch,
                                                                              (epoch - 1) / self.config.max_epoch * 100))
                validation_loss = self.validate()
                vl=np.mean(validation_loss)
                plot_validation_loss.append([epoch, vl])
                if vl < loss_ref:
                    self.logger.info("Best model for now  : saved ")
                    loss_ref = vl
                    self.save_checkpoint(is_best=1)
                torch.cuda.empty_cache()
            self.current_epoch += 1
            if epoch > 1 and epoch >= 2 * self.config.validate_every:
                plt_loss()(plot_training_loss, plot_validation_loss, savefig=join(self.config.out_dir, "loss.png"))
        self.logger.info("Training ended!")


    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.logger.info("training one ep")
        plot_loss = []
        self.model.train()
        batch_idx=0
        for data, target in self.data_loader.train_loader:
            data, mer, coord, target = data[0], data[1], data[2], target[0]
            data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data, mer, coord)
            loss = compute_loss(self.loss,output, target, self.config.loss_weight)
            loss.backward()
            plot_loss.append(loss.item())
            self.optimizer.step()
            batch_idx +=1
            self.current_iteration += 1
        self.logger.info('\t\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \n'.format(
            self.current_epoch, batch_idx * self.config.train_batch_size + len(data),
            len(self.data_loader.train_loader.dataset),
                                (100. * (batch_idx + 1)) / len(self.data_loader.train_loader), loss.item()))
        if self.config.tensorboard:
            self.summary_writer.add_scalars("Loss", {"training_loss": np.mean(plot_loss)}, self.current_epoch)
        self.save_checkpoint()
        return plot_loss


    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        plot_loss = []
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.data_loader.valid_loader:
                data, mer, coord, target = data[0], data[1], data[2], target[0]
                data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(self.device), target.to(self.device)
                output = self.model(data.float(), mer.float(), coord.float())
                l = compute_loss(self.loss,output, target, self.config.loss_weight).item()
                test_loss += l
                plot_loss.append(l)

        self.logger.info(
            '\t\tTest set: Average loss: {:.4f} \n'.format(
                test_loss / len(self.data_loader.valid_loader) ))
        if self.config.tensorboard:
            self.summary_writer.add_scalars("Loss", {"validation_loss": np.mean(plot_loss)}, self.current_epoch)
            out_img = self.data_loader.plot_samples_per_epoch(data, target, output, self.current_epoch)
            self.summary_writer.add_image('train/generated_image', out_img.transpose(2,0,1), self.current_epoch)
        return plot_loss


    def test(self):
        with torch.no_grad():
            ##### Read ground_truth_file
            dic_gt = {}
            with open(self.config.ground_truth_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        dic_gt[row[2].replace(".tif", ".npy")] = row[3]
            self.model = TranslatingUnet(self.config).to(self.device)
            self.load_checkpoint("model_best.pth.tar")
            self.model.eval()

            if self.config.full_test:
                data, target = next(iter(self.data_loader.test_loader))
                data, mer, coord, target = data[0], data[1], data[2], target[0]
                data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(self.device), target.to(self.device)
                _, n_labels_tgt, x_tgt, y_tgt = target.shape
                _, n_labels_src, x_src, y_src = data.shape
                out = torch.zeros(1, n_labels_tgt, x_tgt, y_tgt).float()
                tgt = torch.zeros(1, n_labels_tgt, x_tgt, y_tgt).float()
                name = []

                for data, target in self.data_loader.test_loader:
                    data, mer, coord,n, target = data[0], data[1], data[2],data[3], target[0],
                    data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(self.device), target.to(self.device)
                    output = self.model(data.float(), mer.float(), coord.float())
                    out = torch.cat((out, output.cpu()), 0)
                    tgt = torch.cat((tgt, target.cpu()), 0)
                    name.extend(n)

                out = out[1:]
                tgt = tgt[1:]

                y_pred = torch.argmax(out, dim=1).int()
                y_true = torch.argmax(tgt, dim=1).int()

                y_pred_gt = y_pred[:, 32, 31].numpy()

                y_pred = y_pred.view(-1).detach().cpu().numpy()
                y_true = y_true.view(-1).detach().cpu().numpy()

                report = classification_report(y_true, y_pred)
                confusion = confusion_matrix(y_true, y_pred)
                kappa = cohen_kappa_score(y_true, y_pred)

                epa = EdgePreservationAssessment(out, tgt)
                epi = epa.EPI()

                with open(join(self.config.out_dir, "agreement_between_prediction_and_target.txt"), "a+") as f:
                    f.write(report)
                    f.write(str(kappa))
                    f.write("\n EPI :{} \n".format(epi))
                np.savetxt(join(self.config.out_dir,"confusion_between__between_prediction_and_target.txt"), confusion)

                # name= [basename(val) for sublist in name for val in sublist]
                name = [basename(i) for i in name]
                name = np.array(name)

                y_true = []
                y_pred = []
                for i in range(len(name)):
                    y_true.append(str(dic_gt[name[i]]))
                    y_pred.append(str(self.data_loader.test_tgt_info.id_labels[y_pred_gt[i]]))

                kappa = cohen_kappa_score(y_true, y_pred)
                report = classification_report(y_true, y_pred)
                confusion = confusion_matrix(y_true, y_pred)

                with open(join(self.config.out_dir, "accuracy_assessement_between_prediction_and_GROUNDTRUTH.txt"), "a+") as f:
                    f.write(report)
                    f.write(str(kappa))
                np.savetxt(join(self.config.out_dir,"confusion_between_prediction_and_GROUNDTRUTH.txt"), confusion)

            else:
                data, target = next(iter(self.data_loader.test_loader))
                data, mer, coord, target = data[0], data[1], data[2], target[0]
                data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(self.device), target.to(
                    self.device)
                _, n_labels_tgt, x_tgt, y_tgt = target.shape
                _, n_labels_src, x_src, y_src = data.shape
                out = torch.zeros(1, n_labels_tgt,).float()
                name = []

                for data, target in self.data_loader.test_loader:
                    data, mer, coord, n, target = data[0], data[1], data[2], data[3], target[0],
                    data, mer, coord, target = data.to(self.device), mer.to(self.device), coord.to(
                        self.device), target.to(self.device)
                    output = self.model(data.float(), mer.float(), coord.float())
                    out = torch.cat((out, output.cpu()[:,:,32, 31]), 0)
                    name.extend(n)

                out = out[1:]

                y_pred_gt = torch.argmax(out, dim=1).int().numpy()

                # name = [basename(val) for sublist in name for val in sublist]
                name = [basename(i) for i in name]
                name = np.array(name)

                y_true = []
                y_pred = []
                for i in range(len(name)):
                    y_true.append(str(dic_gt[name[i]]))
                    y_pred.append(str(self.data_loader.test_tgt_info.id_labels[y_pred_gt[i]]))

                kappa = cohen_kappa_score(y_true, y_pred)
                report = classification_report(y_true, y_pred)
                confusion = confusion_matrix(y_true, y_pred)

                with open(join(self.config.out_dir, "accuracy_assessement_between_prediction_and_GROUNDTRUTH.txt"),
                          "a+") as f:
                    f.write(report)
                    f.write(str(kappa))
                np.savetxt(join(self.config.out_dir, "confusion_between_prediction_and_GROUNDTRUTH.txt"), confusion)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        torch.cuda.empty_cache()
        if self.config.tensorboard:
            self.tensorboard_process.kill()
            self.summary_writer.close()
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.data_loader.finalize()
