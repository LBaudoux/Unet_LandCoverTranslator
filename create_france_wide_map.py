import numpy as np
import argparse

import torch

from datasets.landcover_to_landcover import LandcoverToLandcoverDataLoader

from utils.image_type import OSO, CLC
import json
from os.path import join
from graphs.models.translating_unet import TranslatingUnet as net


import gdal

parser = argparse.ArgumentParser('Create a dataset')
parser.add_argument('-d','--out_dir', dest='out_dir', help='output directory', type=str)
parser.add_argument('-c','--config file', dest='conf_file', help='Reference config file', type=str)
parser.add_argument('-m','--model file', dest='model_file', help='Model to load', type=str)
args = parser.parse_args()


out_dir = args.out_dir


with open(args.conf_file) as f:
    config = json.load(f)

device = torch.device("cuda")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict(config)

config.test_batch_size = 1
config.data_folder=config.data_folder.replace("oso_to_clc","oso_to_clc_self_covering")

"""f=config.data_folder
f=os.path.dirname(f).split("/")[-1]
config.data_folder=os.path.join(os.environ.get('TMPDIR'),f)"""

# define data_loader
data_loader = LandcoverToLandcoverDataLoader(config=config,mode="full")

# Get image src and tgt type
src_type, tgt_type = OSO(), CLC(level=2)


m = args.model_file

gt = (51639.68854192982, 100.0, 0.0, 7150786.697153926, 0.0, -100.0)
prj = 'PROJCS["RGF93 / Lambert-93",GEOGCS["RGF93",DATUM["Reseau_Geodesique_Francais_1993",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6171"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4171"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",49],PARAMETER["standard_parallel_2",44],PARAMETER["latitude_of_origin",46.5],PARAMETER["central_meridian",3],PARAMETER["false_easting",700000],PARAMETER["false_northing",6600000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","2154"]]'
width, height = 12681,11915
no_data_value = 0.0

arr_p = np.zeros((1, width, height)) - 1
# arr_t = np.zeros((1, width, height)) - 1

model = net(config)
# checkpoint = torch.load(m,map_location=torch.device('cpu'))
checkpoint = torch.load(m)
model.load_state_dict(checkpoint['state_dict'])

model.to(device)
model.eval()

with torch.no_grad():
    for data, target in data_loader.full_loader:
        data, mer, coord, n, xy, target = data[0], data[1], data[2], data[3], data[4], target[0]
        data, mer, coord, target = data.to(device), mer.to(device), coord.to(device), target.to(device)
        output = model(data, mer, coord)
        ecart_x = int(np.round(abs(xy[0] * 10000 - gt[0])/ 100,0))
        ecart_y = int(np.round(abs(xy[1] * 10000 - gt[3]) / 100,0))

        arr_p[0, ecart_x + 5:ecart_x + 60, ecart_y + 5:ecart_y + 60] = np.transpose(torch.argmax(output, dim=1)[0, 5:60, 5:60].cpu().numpy(), (1, 0))
        # arr_t[0, ecart_x + 5:ecart_x + 60, ecart_y + 5:ecart_y + 60] = np.transpose(torch.argmax(target, dim=1)[0, 5:60, 5:60].cpu().numpy(), (1, 0))

arr_p = np.transpose(arr_p, (0, 2, 1)) + 1
# arr_t = np.transpose(arr_t, (0, 2, 1)) + 1
cmap = tgt_type.cmap
id_labels = tgt_type.id_labels
ct = gdal.ColorTable()
i = 0
for label, color in zip(id_labels, cmap):
    ct.SetColorEntry(i + 1, color)
    i += 1

driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(join(out_dir,"predicted_CLC_2018_map.tif"), np.shape(arr_p)[2], np.shape(arr_p)[1], 1, gdal.GDT_UInt16)
dataset.SetGeoTransform(gt)
dataset.SetProjection(prj)
for i in range(1):
    dataset.GetRasterBand(i + 1).WriteArray(arr_p[i])
    dataset.GetRasterBand(i + 1).SetNoDataValue(no_data_value)
    try:
        dataset.GetRasterBand(i + 1).SetRasterColorTable(ct)
    except:
        pass
dataset.FlushCache()
dataset = None

# driver = gdal.GetDriverByName('GTiff')
# dataset = driver.Create(join(out_dir,"Original_CLC_2018_mapv2_round.tif"), np.shape(arr_t)[2], np.shape(arr_t)[1], 1, gdal.GDT_UInt16)
# dataset.SetGeoTransform(gt)
# dataset.SetProjection(prj)
# for i in range(1):
#     dataset.GetRasterBand(i + 1).WriteArray(arr_t[i])
#     dataset.GetRasterBand(i + 1).SetNoDataValue(no_data_value)
#     try:
#         dataset.GetRasterBand(i + 1).SetRasterColorTable(ct)
#     except:
#         pass
# dataset.FlushCache()
# dataset = None