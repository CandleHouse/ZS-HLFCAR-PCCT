import torch
import torch.nn.functional as F
from torch_mando import *
from crip.postprocess import *
from crip.io import *
from utils import *
import time


def correctRingArtifactInProj_tensor(sgm, sigma, ksize=None):
    import cv2
    import torch.nn.functional as F
    
    def conv(x, kernel):
        x_torch = x.view(1, 1, -1)
        kernel_torch = torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1).to(x_torch.device)
        padding = (len(kernel) - 1) // 2
        result_torch = F.conv1d(x_torch, kernel_torch, padding=padding)
        return result_torch.squeeze()
    
    ksize = ksize or int(2 * np.ceil(2 * sigma) + 1)
    kernel = np.squeeze(cv2.getGaussianKernel(ksize, sigma))

    B, C, H, W = sgm.shape
    sgm = sgm.reshape(B*C, H, W)
    for i in range(B*C):
        Pc = torch.mean(sgm[i], dim=0).detach()
        Rc = conv(Pc, kernel)
        Ec = Pc - Rc
        sgm[i] = sgm[i] - Ec[None, :]
        
    return sgm.reshape(B, C, H, W)


def pyrDown_torch(image):
    B, C, H, W = image.shape
    gaussian_kernel = np.array([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1]).reshape(5,5) / 256
    gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1).to(image.device)
    blurred_image = F.conv2d(image, gaussian_kernel, stride=2, padding=2)
    downsampled_image = blurred_image
    return downsampled_image

    
def cartesian2polar(image):
    M = image.shape[-1]
    theta = torch.linspace(0, np.deg2rad(179), 512)
    r = torch.linspace(0, M-1, M) + 1 - (M + 1) / 2
    theta, r = torch.meshgrid(theta, r, indexing='ij')
    z = torch.polar(r, theta)

    X, Y = z.real.to(image.device), z.imag.to(image.device)
    grid = torch.stack((X / torch.max(X), Y / torch.max(Y)), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    return F.grid_sample(image, grid, align_corners=False) 
    

def polar2cartesian(image):
    image_left, image_right = image.clone(), image.clone()
    image_left[..., :image.shape[-1]//2] = image[..., :image.shape[-1]//2]
    image_right[..., image.shape[-1]//2:] = image[..., image.shape[-1]//2:]
    image_left2right = torch.flip(image_left, dims=[-1])
    image_cat = torch.cat((image_right, image_left2right), dim=-2)
    
    M = image.shape[-1]
    X, Y = torch.arange(M).to(image.device), torch.arange(M).to(image.device)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    X, Y = X - (M-1)/2, Y - (M-1)/2
    r = torch.sqrt(X ** 2 + Y ** 2)
    theta = torch.atan2(Y, X)

    grid = torch.stack((r / (M//2), theta / np.pi), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
    img = F.grid_sample(image_cat, grid, align_corners=False)
    return torch.flip((torch.rot90(img, dims=[-2, -1], k=1)), dims=[-1])

    
def optimization_LFCAR_PCCT(sgm_raw, cfg: MandoFanBeamConfig, CUDA_VISIBLE_DEVICES=0):
    device = torch.device(f'cuda:{CUDA_VISIBLE_DEVICES}' if torch.cuda.is_available() else 'cpu')

    fpj = torch.FloatTensor(sgm_raw[np.newaxis, np.newaxis, :, :]).to(device)
    
    ### initialize parameters and optimizer
    IV_center = torch.zeros((cfg.detEltCount, 4), dtype=torch.float32).to(device)
    IV_center[:, 1] = 1  # first order initialize as 1
    IV_center.requires_grad = True

    optimizer_inner = torch.optim.Adamax([IV_center], lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # flat
    # optimizer_inner = torch.optim.Adagrad([IV_center], lr=0.0002, lr_decay=0, weight_decay=0, initial_accumulator_value=0)  # boundary good

    criterion = torch.nn.MSELoss()
    
    fbp_list = []
    start_time = time.time()
    for iter in range(50):
        IV_det = IV_center
        ### clip and smooth to limit
        with torch.no_grad():
            torch.clamp_(IV_det[:, 3], min=-0.02, max=0.02)      # a^3
            torch.clamp_(IV_det[:, 2], min=-0.15, max=0.15)     # a^2
            torch.clamp_(IV_det[:, 1], min=0.99, max=1.01)    # a
            torch.clamp_(IV_det[:, 0], min=-0.001, max=0.001)  # 1
        
        fpj_comp = fpj**0 * IV_det[:, 0] + fpj * IV_det[:, 1] + fpj**2 * IV_det[:, 2] + fpj**3 * IV_det[:, 3]

        ### recon
        fbp = MandoFanbeamFbp(fpj_comp, cfg)
        
        ### convert to polar coordinate
        fbp_polar_raw = cartesian2polar(fbp)
        
        fbp_d1 = pyrDown_torch(fbp)
        fbp_polar_d1 = cartesian2polar(fbp_d1)
        
        fbp_d2 = pyrDown_torch(fbp_d1)
        fbp_polar_d2 = cartesian2polar(fbp_d2)
        
        # fbp_d3 = pyrDown_torch(fbp_d2)
        # fbp_polar_d3 = cartesian2polar(fbp_d3)
        
        fbp_polar_old = fbp_polar_raw.clone().detach()
        fbp_polar_filter = correctRingArtifactInProj_tensor(fbp_polar_old, 1.0)
        
        loss = None
        weight = [1, 1, 1, 1]
        ### calc loss
        for fbp_index, fbp_polar in enumerate([fbp_polar_raw, fbp_polar_d1, fbp_polar_d2]):
            fbp_polar = correctRingArtifactInProj_tensor(fbp_polar, 1.0)
            output_du = torch.abs(duTensor(fbp_polar, device))
            output_du2 = torch.abs(ScharrXTensor3(fbp_polar, device))

            boundary = 1  # pixel for roi
            diff_du = output_du[..., boundary: -boundary]
            diff_du2 = output_du2[..., boundary: -boundary]

            if iter == 0:
                diff_du_max = torch.max(diff_du).detach() * diff_du.numel()
                diff_du2_max = torch.max(diff_du2).detach() * diff_du2.numel()
            
            loss_ = criterion(300 * torch.sum(diff_du) / diff_du_max + 
                             700 * torch.sum(diff_du2) / diff_du2_max,
                             torch.tensor(0, dtype=torch.float32).to(device)) \
                             * weight[fbp_index]
            loss = loss_ if loss is None else loss + loss_
            
        print(loss.item())
        
        optimizer_inner.zero_grad()
        loss.backward()
        optimizer_inner.step()
        
        fbp = polar2cartesian(fbp_polar_filter)
        fbp_list.append(take(fbp.squeeze()))
    
    print('Total cost:', time.time() - start_time)
    return np.array(fbp_list, dtype=np.float32)
    
    
def optimization_LFCAR_PCCT_dual(sgm, cfg: MandoFanBeamConfig, CUDA_VISIBLE_DEVICES=0):
    device = torch.device(f'cuda:{CUDA_VISIBLE_DEVICES}' if torch.cuda.is_available() else 'cpu')
    
    def optimization_LFCAR_PCCT_once(x, y, fpj):
        IV_center = torch.zeros((cfg.detEltCount, 6), dtype=torch.float32).to(device)
        IV_center[:, 1] = x  # first order initialize as 1
        IV_center[:, 2] = y  # first order initialize as 1
        IV_center.requires_grad = True

        optimizer_inner = torch.optim.Adagrad([IV_center], lr=0.0002, lr_decay=0, weight_decay=0, initial_accumulator_value=0)  # boundary good
        criterion1 = torch.nn.MSELoss()
    
        fbp_list = []
        for iter in range(50):
            IV_det = IV_center

            fpj_1, fpj_2 = fpj[:, 0][:, None], fpj[:, 1][:, None]
            fpj_comp = fpj_1**0 * IV_det[:, 0] + fpj_1 * IV_det[:, 1] + fpj_2 * IV_det[:, 2] + \
                        fpj_1**2 * IV_det[:, 3] + fpj_2**2 * IV_det[:, 4] + fpj_1 * fpj_2 * IV_det[:, 5]

            ### recon
            fbp = MandoFanbeamFbp(fpj_comp, cfg)
            
            ### convert to polar coordinate
            fbp_polar_raw = cartesian2polar(fbp)
            
            fbp_d1 = pyrDown_torch(fbp)
            fbp_polar_d1 = cartesian2polar(fbp_d1)
            
            fbp_d2 = pyrDown_torch(fbp_d1)
            fbp_polar_d2 = cartesian2polar(fbp_d2)
            
            # fbp_d3 = pyrDown_torch(fbp_d2)
            # fbp_polar_d3 = cartesian2polar(fbp_d3)
            
            # if iter == 0:
            fbp_polar_old = fbp_polar_raw.clone().detach()
            fbp_polar_filter = correctRingArtifactInProj_tensor(fbp_polar_old, 1.0)

            loss = None
            weight = [1, 0.5, 0.1, 1]
            ### calc loss
            for fbp_index, fbp_polar in enumerate([fbp_polar_raw, fbp_polar_d1, fbp_polar_d2]):
                
                fbp_polar = correctRingArtifactInProj_tensor(fbp_polar, 1.0)
                diff_du = torch.abs(duTensor(fbp_polar, device))
                diff_du2 = torch.abs(ScharrXTensor3(fbp_polar, device))
                
                loss_ = criterion1(3 * torch.mean(diff_du) + 
                                7 * torch.mean(diff_du2),
                                torch.tensor(0, dtype=torch.float32).to(device)) \
                                * weight[fbp_index]

                loss = loss_ if loss is None else loss + loss_
            
            print(loss.item())
            
            optimizer_inner.zero_grad()
            loss.backward()
            optimizer_inner.step()

            fbp = polar2cartesian(fbp_polar_filter)
            fbp_list.append(take(fbp.squeeze()))
                
        return fbp
    
    fpj = torch.FloatTensor(sgm[np.newaxis, :, :, :]).to(device)
    fbp_2xy = optimization_LFCAR_PCCT_once(2, 1, fpj)
    fbp_x2y = optimization_LFCAR_PCCT_once(1, 2, fpj)
    fbp_x = fbp_2xy - (fbp_2xy + fbp_x2y) / 3
    fbp_y = fbp_x2y - (fbp_2xy + fbp_x2y) / 3
    
    return take(fbp_x), take(fbp_y)