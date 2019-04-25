import torch
import torch.nn.functional as F


class OctConv(torch.nn.Module):
  def __init__(self, ch_in, ch_out, kernel_size, 
               stride=1, padding=0, dilation=1, 
               groups=1, bias=True, 
               alphas=(0.5,0.5)):
    super(OctConv, self).__init__()
    assert ch_in > 1 and ch_out > 1, "channels must be larger than 1, (1, ~)"
    self.alpha_in, self.alpha_out = alphas
    assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_out <= 1, "Alphas must be in interval [0, 1]"

    self.lf_in = max(1, int(self.alpha_in * ch_in))
    self.hf_in = ch_in - self.lf_in
    self.lf_out = max(1, int(self.alpha_out * ch_out))
    self.hf_out = ch_out - self.lf_out

    print('\nalpha: ', (self.alpha_in, self.alpha_out), 
          '\nch_in_hf: ', self.hf_in, 
          '\nch_in_lf: ', self.lf_in, 
          '\nch_out_hf: ', self.hf_out, 
          '\nch_out_lf: ', self.lf_out, 
          '\npadding: ', padding, 
          '\nstride: ', stride, 
          '\nkernel_size: ', kernel_size)

    self.L2L = torch.nn.Conv2d(self.lf_in, self.lf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.L2H = torch.nn.Conv2d(self.lf_in, self.hf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.H2L = torch.nn.Conv2d(self.hf_in, self.lf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.H2H = torch.nn.Conv2d(self.hf_in, self.hf_out, kernel_size, stride, padding, dilation, groups, bias)
    self.layers = [self.L2L, self.L2H, self.H2L, self.H2H]
    return

  def upscale(self, x, scale = 2):
    return F.interpolate(x, scale_factor = scale, mode = 'nearest')

  def downscale(self, x, scale = 2):
    return F.avg_pool2d(x, scale, stride = scale)

  def forward(self, x):
    x_lf = self.downscale(x[:, :self.lf_in, ...])
    LtoL = self.L2L(x_lf)
    LtoH = self.upscale(self.L2H(x_lf))
    x_hf = x[:, self.lf_in:, ...]
    HtoH = self.H2H(x_hf)
    HtoL = self.H2L(self.downscale(x_hf))
    output = torch.cat([self.upscale(LtoL + HtoL), (LtoH + HtoH)], 1)
    return output


if __name__ == '__main__':
  ch_in, ch_out = 3, 12

  oct = OctConv(ch_in, ch_out, 3, alphas = (0.2, 0.2))
  x = torch.rand([2, ch_in, 32, 32])

  print(oct(x).shape)
