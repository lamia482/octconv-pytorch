import torch
import torch.nn.functional as F
from torch.nn import Conv2d
from logger import Logger


logger = Logger(__name__).get()


class OctConv(torch.nn.Module):
  def __init__(self, ch_in, ch_out, kernel_size, 
               stride = 1, padding = 0, dilation = 1, 
               groups = 1, bias = True, 
               alphas = (0.3, 0.3)):
    super(OctConv, self).__init__()
    assert ch_in > 1 and ch_out > 1, "channels must be larger than 1, (1, ~)"
    self.alpha_in, self.alpha_out = alphas
    assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_out <= 1, "Alphas must be in interval [0, 1]"

    self.lf_in = max(1, int(self.alpha_in * ch_in))
    self.hf_in = ch_in - self.lf_in
    self.lf_out = max(1, int(self.alpha_out * ch_out))
    self.hf_out = ch_out - self.lf_out

    self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    self.X2L = Conv2d(ch_in, self.lf_in, kernel_size, 1, padding, dilation, groups, bias)
    self.X2H = Conv2d(ch_in, self.hf_in, kernel_size, 1, padding, dilation, groups, bias)
    self.L2L = Conv2d(self.lf_in, self.lf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.L2H = Conv2d(self.lf_in, self.hf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.H2L = Conv2d(self.hf_in, self.lf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.H2H = Conv2d(self.hf_in, self.hf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.L2X = Conv2d(self.lf_out, ch_out, kernel_size, 1, padding, dilation, groups, bias)
    self.H2X = Conv2d(self.hf_out, ch_out, kernel_size, 1, padding, dilation, groups, bias)
    self.layers = [self.X2L, self.X2H, self.L2L, self.L2H, self.H2L, self.H2H, self.L2X, self.H2X]
    return

  def upscale(self, x, scale = 2):
    return F.interpolate(x, scale_factor = scale, mode = 'nearest')

  def downscale(self, x, scale = 2):
    return F.avg_pool2d(x, scale, stride = scale)

  def forward_beta(self, x):
    x_lf = self.X2L(self.downscale(x))
    LtoL = self.L2L(x_lf)
    LtoH = self.upscale(self.L2H(x_lf))
    x_hf = self.X2H(x)
    HtoL = self.H2L(self.downscale(x_hf))
    HtoH = self.H2H(x_hf)
    L = LtoL + HtoL
    H = LtoH + HtoH
    output = self.upscale(self.L2X(L)) + self.H2X(H)
    # logger.debug('Receive {} --> {} [{}, {}, {}]'.format(x.shape, output.shape, self.kernel_size, self.stride, self.padding))
    return output

  def forward(self, x):
    return self.forward_beta(x)
    # return self.forward_desperated(x)
    x_lf = x[:, :self.lf_in, ...]
    LtoL = self.L2L(x_lf)
    LtoH = self.L2H(x_lf)
    x_hf = x[:, self.lf_in:, ...]
    HtoH = self.H2H(x_hf)
    HtoL = self.H2L(x_hf)
    output = torch.cat([LtoL + HtoL, LtoH + HtoH], 1)
    return output

  def forward_desperated(self, x):
    x_lf = self.downscale(x[:, :self.lf_in, ...])
    LtoL = self.L2L(x_lf)
    LtoH = self.upscale(self.L2H(x_lf))
    x_hf = x[:, self.lf_in:, ...]
    HtoH = self.H2H(x_hf)
    HtoL = self.H2L(self.downscale(x_hf))
    Ls = LtoL + HtoL
    L = self.upscale(Ls)
    H = LtoH + HtoH
    output = torch.cat([L, H], 1)
    return output


if __name__ == '__main__':
  ch_in, ch_out = 3, 12

  oct = OctConv(ch_in, ch_out, 3, alphas = (0.2, 0.2))
  x = torch.rand([2, ch_in, 32, 32])

  print(oct(x).shape)
