import torch
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3: return R.from_euler('xyz', r.squeeze(), degrees=False).as_matrix()
    else: return R.from_matrix(r).as_euler('xyz', degrees=False).reshape((3, 1))

def sqrt(x):
    if type(x) == torch.Tensor: return torch.sqrt(x)
    else: return np.sqrt(x)

def atan2(y, x):
    if type(x) == torch.Tensor: return torch.atan2(y, x)
    else: return np.arctan2(y, x)

def asin(x):
    if type(x) == torch.Tensor: return torch.asin(x)
    else: return np.arcsin(x)

def acos(x):
    if type(x) == torch.Tensor: return torch.acos(x)
    else: return np.arccos(x)

def cos(x):
    if type(x) == torch.Tensor: return torch.cos(x)
    else: return np.cos(x)

def sin(x):
    if type(x) == torch.Tensor: return torch.sin(x)
    else: return np.sin(x)

def exp(x):
    if type(x) == torch.Tensor: return torch.exp(x)
    else: return np.exp(x)

def reshape(x, shape):
    if type(x) == torch.Tensor: return x.view(shape)
    else: return x.reshape(shape)

def toNumpy(arr) -> np.ndarray:
    if type(arr) == torch.Tensor: arr = arr.cpu().numpy()
    return arr

def concat(arr_list: list, axis=0):
    if type(arr_list[0]) == torch.Tensor: return torch.cat(arr_list, dim=axis)
    else: return np.concatenate(arr_list, axis=axis)

def polyval(P, x):
    if type(x) == torch.Tensor: P = torch.Tensor(P).to(x.device)
    if type(P) == torch.Tensor:
        npol = P.shape[0]
        val = torch.zeros_like(x)
        for i in range(npol-1):
            val = val * x + P[i] * x
        val += P[-1]
        return val
    else:
        return np.polyval(P, x)

def pixelToGrid(pts, target_resolution: (int, int), 
                source_resolution: (int, int)):
    h, w = target_resolution
    height, width = source_resolution
    xs = (pts[0,:]) / (width - 1) * 2 - 1
    ys = (pts[1,:]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)

class FisheyeCamModel:
    def __init__(self): pass
    def setConfig(self, cfg):
        self.height, self.width = cfg['image_size']
        self.max_theta = cfg['max_fov']
        self.panow = int(self.width * 360 / self.max_theta / 2 * 2)
        self.panoh = int(self.panow * 0.5)
        self.valid_height, self.valid_width = cfg['valid_size']

        self.cx1, self.cy1 = cfg['center1']
        self.R1 = np.array(cfg['R1']).reshape(3,3)
        self.T1 = np.array(cfg['T1']).reshape(3,1)
        self.f1 = cfg['f1']
        self.k1 = cfg['k1']

        self.cx2, self.cy2 = cfg['center2']
        self.R2 = np.array(cfg['R2']).reshape(3,3)
        self.T2 = np.array(cfg['T2']).reshape(3,1)
        self.f2 = cfg['f2']
        self.k2 = cfg['k2']
        self.R_x_w = self.GetStereoR()

    # end setConfig

    def GetStereoR(self):
        t12 = np.matmul(-self.R2.transpose(), self.T2)
        norm_t12 = t12 / LA.norm(t12)
        rvec = np.zeros(norm_t12.shape)
        rvec[1] = -norm_t12[2]
        rvec[2] = norm_t12[1]
        rvec = rvec * asin(LA.norm(rvec)) / (LA.norm(rvec) + 1e-9)
        R_x_w = rodrigues(rvec)
        return R_x_w

    def UndisortTheta(self, theta, coeffs):
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        res = theta + coeffs[1] * theta2 + coeffs[2] * theta3 + coeffs[3] * theta4
        return res

    def getDeriv(self, theta, distort):
        deriv = distort[0]
        curTheta = 1.0
        for i in range(1, len(distort)):
            curTheta = curTheta * theta
            if distort[i] != 0:
                deriv = deriv + distort[i] * curTheta
        return deriv

    def getThetaInDistort(self, radius, focal, distort):
        curTheta = np.pi / 2
        cnt = 0
        maxCnt = 50
        diff = 0
        while cnt < maxCnt:
            curR = focal * self.UndisortTheta(curTheta, distort)
            diff = curR - radius
            delta = -diff / (focal * self.getDeriv(curTheta, distort))
            curTheta = curTheta + delta
            cnt = cnt + 1
            if np.abs(diff) < 0.01:
                return curTheta
        return -1

    def getDeriv1(self, theta, distort):
        deriv = distort[0]
        curTheta = 1.0
        for i in range(1, len(distort)):
            curTheta = curTheta * theta
            if distort[i] != 0:
                deriv = deriv + (i+1) * distort[i] * curTheta # * (i+1)
        return deriv

    def getThetaInDistort1(self, radius, focal, distort):
        curTheta = radius/focal # better than pi/2
        cnt = 0
        maxCnt = 50
        while cnt < maxCnt:
            curR = focal * self.UndisortTheta(curTheta, distort)
            diff = curR - radius
            delta = -diff / (focal * self.getDeriv(curTheta, distort))
            curTheta = curTheta + delta
            cnt = cnt + 1
            if np.abs(diff) < 0.0001:
                return curTheta
        return -1

    def pano2fisheye(self, f, cx, cy, k):
        upano, vpano = np.meshgrid(range(self.panow), range(self.panoh))
        Phi = np.pi * 2 - np.pi * 2 * upano / self.panow
        theta = np.pi * vpano / self.panoh

        Phi = Phi.reshape(1, -1)
        theta = theta.reshape(1, -1)

        undistort_theta = self.UndisortTheta(theta, k)
        rho = f * undistort_theta

        u = rho * cos(Phi) + cx
        v = rho * sin(Phi) + cy

        u = u.reshape(1, -1) / self.width
        v = v.reshape(1, -1) / self.height
        out = concat((u, v), axis=0)

        theta = theta.reshape(1, -1)

        out[:, theta.squeeze() < 0] = 0
        out[:, theta.squeeze() > self.max_theta * 0.5 / 180.0 * np.pi] = 0

        return out

    def fisheye2sphere(self, f, cx, cy, k):
        max_fov_rad = self.max_theta * 0.5 / 180.0 * np.pi
        max_r = f * self.UndisortTheta(max_fov_rad, k)

        x, y = np.meshgrid(range(self.width), range(self.height))
        y = y.reshape(1,-1) - cy
        x = x.reshape(1,-1) - cx
        rho = sqrt(x * x + y * y)

        theta = []
        for i in range(0, rho.shape[1]):
            if rho[0,i] > max_r:
                theta.append(self.max_theta / 180.0 * np.pi)
                continue
            theta.append(self.getThetaInDistort(rho[0,i], f, k))
        theta = (np.array(theta)).reshape(1, -1)

        fi = atan2(y, x)
        x1 = sin(theta) * cos(fi)
        y1 = sin(theta) * sin(fi)
        z1 = cos(theta)

        out = concat((x1, y1, z1), axis=0)

        out[:, theta.squeeze() >= self.max_theta * 0.5 / 180.0 * np.pi] = 0

        return out

    def fisheye2verticalpano(self, f, cx, cy, k, panowidth, panoheight, fov, R):
        max_fov_rad = fov / 180.0 * np.pi
        upano, vpano = np.meshgrid(range(panowidth), range(panoheight))
        angle_x = (1 - (upano + 0.5) / panowidth) * max_fov_rad + (np.pi - max_fov_rad) * 0.5
        angle_yz = ((vpano + 0.5) / panoheight - 0.5) * max_fov_rad

        angle_x = angle_x.reshape(1, -1)
        angle_yz = angle_yz.reshape(1, -1)

        x1 = cos(angle_x)
        y1 = sin(angle_x) * sin(angle_yz)
        z1 = sin(angle_x) * cos(angle_yz)

        P1 = concat([x1, y1, z1], axis=0)
        P2 = np.matmul(R, P1)

        invdepth = 1.0 / sqrt(P2[0,:] * P2[0,:] + P2[1,:] * P2[1,:] + P2[2,:] * P2[2,:])
        x2 = P2[0,:] * invdepth
        y2 = P2[1,:] * invdepth
        z2 = P2[2,:] * invdepth

        theta = acos(z2)
        fi = atan2(y2, x2)

        undistort_theta = self.UndisortTheta(theta, k)
        rho = f * undistort_theta

        u = rho * cos(fi) + cx
        v = rho * sin(fi) + cy

        u = u.reshape(1, -1)
        v = v.reshape(1, -1)
        out = concat((u, v), axis=0)

        return out

    def verticalpano2fisheye(self, f, cx, cy, k, outwidth, outheight, srcwidth, srcheight, fov, R):
        max_fov_rad = fov / 180.0 * np.pi
        max_fov_rad_2 = fov * 0.5 / 180.0 * np.pi
        max_r = f * self.UndisortTheta(max_fov_rad_2, k)

        x, y = np.meshgrid(range(outwidth), range(outheight))
        y = y.reshape(1,-1) - cy
        x = x.reshape(1,-1) - cx
        rho = sqrt(x * x + y * y)

        theta = []
        for i in range(0, rho.shape[1]):
            if rho[0,i] > max_r:
                theta.append(fov / 180.0 * np.pi)
                continue
            theta.append(self.getThetaInDistort(rho[0,i], f, k))
        theta = (np.array(theta)).reshape(1, -1)
        fi = atan2(y, x)
        x1 = sin(theta) * cos(fi)
        y1 = sin(theta) * sin(fi)
        z1 = cos(theta)

        P1 = concat([x1, y1, z1], axis=0)
        P2 = np.matmul(R, P1)

        invdepth = 1.0 / sqrt(P2[0,:] * P2[0,:] + P2[1,:] * P2[1,:] + P2[2,:] * P2[2,:])
        x2 = P2[0,:] * invdepth
        y2 = P2[1,:] * invdepth
        z2 = P2[2,:] * invdepth

        angle_x = acos(x2)
        angle_yz = atan2(y2,z2)

        u = (1.0 - (angle_x - 0.5 * (np.pi - max_fov_rad)) / max_fov_rad) * srcwidth
        v = (angle_yz / max_fov_rad + 0.5) * srcheight

        u = u.reshape(1, -1)
        v = v.reshape(1, -1)

        out = concat((u, v), axis=0)
        out[:, theta.squeeze() >= fov * 0.5 / 180.0 * np.pi] = -100

        out[:, angle_x <= 20/180*np.pi] = -100
        out[:, angle_x >= 160/180*np.pi] = -100

        return out

    def compute_viewmap(self, R, T, depth, fish2Spmap, Sp2fishmap):
        P = np.matmul(R, fish2Spmap) * depth + T
        x = (P[0,:]).reshape(1,-1)
        y = (P[1,:]).reshape(1,-1)
        z = (P[2,:]).reshape(1,-1)
        invdepth = 1.0 / sqrt(x * x + y * y + z * z)
        x = x * invdepth
        y = y * invdepth
        z = z * invdepth

        theta = acos(z)
        phi = atan2(y,x)

        u = np.zeros(theta.shape)
        v = np.zeros(theta.shape)

        spx = fish2Spmap[0,:]
        spy = fish2Spmap[1,:]
        spz = fish2Spmap[2,:]

        for i in range(theta.shape[1]):
            if spx[i] < -0.9 and spy[i] < -0.9 and spz[i] < -0.9:
                continue
            
            uPano = (np.pi * 2 - phi[:,i]) * self.panow / (np.pi * 2) if phi[:,i] > 0.0 \
                    else -phi[:,i] * self.panow / (np.pi * 2)
            vPano = theta[:,i] * self.panoh / np.pi

            uPano0 = int(uPano)
            vPano0 = int(vPano)

            k00 = vPano0 * self.panow + uPano0
            k01 = k00 + 1
            k10 = k00 + self.panow
            k11 = k10 + 1

            uc1 = uPano - float(uPano0)
            vc1 = vPano - float(vPano0)
            c00 = (1.0 - uc1) * (1.0 - vc1)
            c01 = uc1 * (1.0 - vc1)
            c10 = (1.0 - uc1) * vc1
            c11 = uc1 * vc1

            u[:,i] = (Sp2fishmap[0,k00] * c00 + Sp2fishmap[0,k01] * c01 +
                Sp2fishmap[0,k10] * c10 + Sp2fishmap[0,k11] * c11) * self.width
            v[:,i] = (Sp2fishmap[1,k00] * c00 + Sp2fishmap[1,k01] * c01 +
                Sp2fishmap[1,k10] * c10 + Sp2fishmap[1,k11] * c11) * self.height
            u[:,i] = max(min(u[:,i], self.width - 1.0), 0.0)
            v[:,i] = max(min(v[:,i], self.height - 1.0), 0.0)

        u = u.reshape(1, -1)
        v = v.reshape(1, -1)

        out = concat((u, v), axis=0)
        return out

    def calc_left_pano2fisheye(self):
        return self.pano2fisheye(self.f1, self.cx1, self.cy1, self.k1)

    def calc_right_pano2fisheye(self):
        return self.pano2fisheye(self.f2, self.cx2, self.cy2, self.k2)

    def calc_left_fisheye2sphere(self):
        return self.fisheye2sphere(self.f1, self.cx1, self.cy1, self.k1)

    def calc_right_fisheye2sphere(self):
        return self.fisheye2sphere(self.f2, self.cx2, self.cy2, self.k2)

    def calc_l2r_viewmap(self, depth, fish2Spmap, Sp2fishmap):
        return self.compute_viewmap(self.R2, self.T2, depth, fish2Spmap, Sp2fishmap)

    def calc_left_valid_fisheye(self):
        return self.getGridFisheyeValid(self.f1, self.k1, self.cx1, self.cy1)

    def calc_right_valid_fisheye(self):
        return self.getGridFisheyeValid(self.f2, self.k2, self.cx2, self.cy2)

    def calc_left_inv_valid_fisheye(self):
        return self.getGridInvFisheyeValid(self.f1, self.k1, self.cx1, self.cy1)

    def calc_right_inv_valid_fisheye(self):
        return self.getGridInvFisheyeValid(self.f2, self.k2, self.cx2, self.cy2)

    def calc_left_verticalpano2fisheye(self, src_width, src_height, fov, ratio=1.0):
        R_1_w = np.matmul(self.R1, self.R_x_w)
        return self.verticalpano2fisheye(self.f1*ratio, self.cx1*ratio, self.cy1*ratio, self.k1,
                                          int(self.width*ratio), int(self.height*ratio), src_width, src_height, fov, R_1_w.transpose())

    def calc_right_verticalpano2fisheye(self, src_width, src_height, fov):
        R_2_w = np.matmul(self.R2, self.R_x_w)
        return self.verticalpano2fisheye(self.f2, self.cx2, self.cy2, self.k2, self.width, self.height, src_width, src_height, fov, R_2_w.transpose())

    def calc_left_fisheye2verticalpano(self, outwidth, outheight, fov):
        R_1_w = np.matmul(self.R1, self.R_x_w)
        return self.fisheye2verticalpano(self.f1, self.cx1, self.cy1, self.k1, outwidth, outheight, fov, R_1_w)

    def calc_right_fisheye2verticalpano(self, outwidth, outheight, fov):
        R_2_w = np.matmul(self.R2, self.R_x_w)
        return self.fisheye2verticalpano(self.f2, self.cx2, self.cy2, self.k2, outwidth, outheight, fov, R_2_w)

    def getGridInvFisheyeValid(self, f, k, cx, cy):
        max_fov_rad = self.max_theta * 0.5 / 180.0 * np.pi
        max_r = f * self.UndisortTheta(max_fov_rad, k)

        x, y = np.meshgrid(range(self.width), range(self.height))
        y = y.reshape(1,-1) - cy
        x = x.reshape(1,-1) - cx
        rho = sqrt(x * x + y * y)

        inv_valid_x = np.zeros((self.height, self.width), dtype = np.float32)

        inv_valid_y = np.zeros((self.height, self.width), dtype = np.float32)
        inv_valid_x = inv_valid_x.reshape(1,-1)
        inv_valid_y = inv_valid_y.reshape(1,-1)
        j = 0
        for i in range(0, rho.shape[1]):
            if rho[0,i] > max_r:
                continue
            theta = self.getThetaInDistort(rho[0,i], f, k)
            if theta >= self.max_theta * 0.5 / 180.0 * np.pi:
                continue
            inv_valid_x[:, i] = float(j % self.valid_width)
            inv_valid_y[:, i] = float(int(j / self.valid_width))
            j = j + 1
        out = concat((inv_valid_x, inv_valid_y), axis=0)

        return out

    def getGridFisheyeValid(self, f, k, cx, cy):
        max_fov_rad = self.max_theta * 0.5 / 180.0 * np.pi
        max_r = f * self.UndisortTheta(max_fov_rad, k)

        x, y = np.meshgrid(range(self.width), range(self.height))
        y = y.reshape(1,-1)
        x = x.reshape(1,-1)
        uy = y - cy
        ux = x - cx
        rho = sqrt(ux * ux + uy * uy)

        valid_x = np.zeros((self.valid_height, self.valid_width), dtype = np.float32)
        valid_y = np.zeros((self.valid_height, self.valid_width), dtype = np.float32)
        valid_x = valid_x.reshape(1,-1)
        valid_y = valid_y.reshape(1,-1)
        cnt = 0
        for i in range(0, rho.shape[1]):
            if rho[0,i] > max_r:
                continue
            theta = self.getThetaInDistort(rho[0,i], f, k)
            if theta >= self.max_theta * 0.5 / 180.0 * np.pi:
                continue
            valid_x[:,cnt] = x[0,i]
            valid_y[:,cnt] = y[0,i]
            cnt = cnt + 1

        out = concat((valid_x, valid_y), axis=0)
        
        return out
