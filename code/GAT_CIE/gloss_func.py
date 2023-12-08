import torch
import math
import gpytorch

class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):
        # L_X = torch.matmul(X, X.T)
        # L_Y = torch.matmul(Y, Y.T)
        # L_X = gpytorch.kernels.Linear
        covar_module = gpytorch.kernels.LinearKernel().cuda()
        L_X = covar_module(X).to_dense()
        L_Y = covar_module(Y).to_dense()
        return torch.sum(self.centering(L_X) * self.centering(L_Y))
    
    def rbf_HSIC(self, X, Y, sigma):
        covar_module = gpytorch.kernels.RBFKernel().cuda()
        L_X = covar_module(X).to_dense()
        L_Y = covar_module(Y).to_dense()
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def poly_HSIC(self, X, Y, p=2):
        covar_module = gpytorch.kernels.PolynomialKernel(power=2).cuda()
        L_X = covar_module(X).to_dense()
        L_Y = covar_module(Y).to_dense()
        return torch.sum(self.centering(L_X) * self.centering(L_Y))
    
    def rq_HSIC(self, X, Y):
        covar_module = gpytorch.kernels.RQKernel().cuda()
        L_X = covar_module(X).to_dense()
        L_Y = covar_module(Y).to_dense()
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X) + 1e-4)
        var2 = torch.sqrt(self.linear_HSIC(Y, Y) + 1e-4)
        # if hsic / (var1*var2).item() < 0:
        #     assert False
        return hsic / (var1 * var2)

    def rbf_CKA(self, X, Y, sigma=None):
        hsic = self.rbf_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.rbf_HSIC(X, X, sigma) + 1e-4)
        var2 = torch.sqrt(self.rbf_HSIC(Y, Y, sigma) + 1e-4)
        return hsic / (var1 * var2)
    
    def poly_CKA(self, X, Y):
        hsic = self.poly_HSIC(X, Y)
        var1 = torch.sqrt(self.poly_HSIC(X, X) + 1e-4)
        var2 = torch.sqrt(self.poly_HSIC(Y, Y) + 1e-4)
        # if hsic / (var1*var2).item() < 0:
        #     assert False
        return hsic / (var1 * var2)

    def rq_CKA(self, X, Y):
        hsic = self.rq_HSIC(X, Y)
        var1 = torch.sqrt(self.rq_HSIC(X, X) + 1e-4)
        var2 = torch.sqrt(self.rq_HSIC(Y, Y) + 1e-4)
        # if hsic / (var1*var2).item() < 0:
        #     assert False
        return hsic / (var1 * var2)
