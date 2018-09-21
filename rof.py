# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:37:00 2018

@author: zyz
"""

from numpy import *
import numpy as np
#  https://blog.csdn.net/u013378306/article/details/70156842 
def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    m,n = im.shape # 噪声图像的大小
    # 初始化
    U = U_init
    Px = im # 对偶域的x 分量
    Py = im # 对偶域的y 分量
    error = 1 
    while (error > tolerance):
        Uold = U 
          # 原始变量的梯度
        # 原始变量的梯度
        GradUx = np.roll(U,-1,axis=1)-U # 变量U 梯度的x 分量
        GradUy = np.roll(U,-1,axis=0)-U # 变量U 梯度的y 分量 
        # 更新对偶变量
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2)) 
        Px = PxNew/NormNew # 更新x 分量（对偶）
        Py = PyNew/NormNew # 更新y 分量（对偶） 
        # 更新原始变量
        RxPx = np.roll(Px,1,axis=1) # 对x 分量进行向右x 轴平移
        RyPy = np.roll(Py,1,axis=0) # 对y 分量进行向右y 轴平移 
        DivP = (Px-RxPx)+(Py-RyPy) # 对偶域的散度
        U = im + tv_weight*DivP # 更新原始变量 
        # 更新误差
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m);
    return U,im-U # 去噪后的图像和纹理残余 