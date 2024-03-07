import torch
from torch_mando import *
from crip.io import *
from crip.preprocess import *
import numpy as np
from LFCAR import *
import kornia


SID = 100  # mm
SDD = SID * 2  # mm
NView = 600 * 4
DetectorPitch = 0.2  # mm
Width = 384
Height = 64
VolH, VolW, VolN = 512, 512, 96
VoxelSize = 0.075  # mm
detOffCenter = -1.3  # mm
recon_type = 'fbp'


def gen_tradition(proj):
    # raw
    sgm = torch.FloatTensor(proj.copy()[np.newaxis, np.newaxis, :, :]).cuda()
    with torch.no_grad():
        rec = MandoFanbeamFbp(sgm, cfg)
    imwriteTiff(take(rec), './raw.tif')
    
    # sinogram domain smooth
    sgm = correctRingArtifactInProj(proj.copy(), 1.5)
    sgm = torch.FloatTensor(sgm[np.newaxis, np.newaxis, :, :]).cuda()
    with torch.no_grad():
        rec = MandoFanbeamFbp(sgm, cfg)
    imwriteTiff(take(rec), './SDS.tif')
    
    # polar domain smooth
    sgm = torch.FloatTensor(proj.copy()[np.newaxis, np.newaxis, :, :]).cuda()
    with torch.no_grad():
        rec = MandoFanbeamFbp(sgm, cfg)
    fbp_polar = cartesian2polar(rec)
    fbp_polar_filter = kornia.filters.box_blur(fbp_polar, (1, 5))
    rec = polar2cartesian(fbp_polar_filter)
    imwriteTiff(take(rec), './PDS.tif')

    
if __name__ == '__main__':
    
    sgm_le = imreadRaw('./sgm/rat/sgm_low.raw', h=NView, w=Width, nSlice=Height, dtype=np.float32)
    sgm_he = imreadRaw('./sgm/rat/sgm_high.raw', h=NView, w=Width, nSlice=Height, dtype=np.float32)
    cfg = MandoFanBeamConfig(imgDim=VolH, pixelSize=VoxelSize, sid=SID, sdd=SDD, detEltCount=Width, 
                            detEltSize=DetectorPitch, views=NView, reconKernelEnum=KERNEL_GAUSSIAN_RAMP,
                            reconKernelParam=0.5, detOffCenter=detOffCenter, totalScanAngle=-360, imgRot=270)
    slice = Height // 2
    sgm_le_slice = np.mean(np.array([sgm_le[slice-1], sgm_le[slice], sgm_le[slice+1]]), axis=0)
    sgm_he_slice = np.mean(np.array([sgm_he[slice-1], sgm_he[slice], sgm_he[slice+1]]), axis=0)
    
    # single energy version
    fbp_list1 = optimization_LFCAR_PCCT(sgm_le_slice, cfg, 0)
    fbp_list2 = optimization_LFCAR_PCCT(sgm_he_slice, cfg, 0)
    
    # dual energy version
    img1, img2 = optimization_LFCAR_PCCT_dual(np.array([sgm_le_slice, sgm_he_slice]), cfg, 0)

