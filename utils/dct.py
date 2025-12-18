import torch
import numpy as np
__all__ = ["DCT_TORCH"]

class DCT_TORCH:
    def __init__(self, span, dct_stats=None, root=".", dtype=torch.float32, device='cpu', DCT_MAT_SIZE=8, mask='weak', img_size=32, batch_size=512, rgb = False):
        
        self.span = span
        self.root = root
        self.dtype = dtype
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.DCT_MAT_SIZE = DCT_MAT_SIZE
        self.rgb = rgb
        
        self.load_masks(mask)
        
        if dct_stats:
            self.mu, self.std = dct_stats['mean'].reshape(3, 32, 32).unsqueeze(0).mean([2, 3], keepdim=True).to(device), dct_stats['std'].reshape(3, 32, 32).unsqueeze(0).mean([2, 3], keepdim=True).to(device)
        else:
            self.mu, self.std = None, None
        
        DCT_MAT = self.create_dct_matrix(DCT_MAT_SIZE)
        DCT_T = DCT_MAT.T
        self.DCT_MAT = DCT_MAT.unsqueeze(0).expand(batch_size * ((img_size//DCT_MAT_SIZE)**2) * 3, DCT_MAT_SIZE, DCT_MAT_SIZE).to(device)
        self.DCT_T = DCT_T.unsqueeze(0).expand(batch_size * ((img_size//DCT_MAT_SIZE)**2) * 3, DCT_MAT_SIZE, DCT_MAT_SIZE).to(device)
        
    
    def load_masks(self, mask):
        """
        Load the masks for the luminance and chrominance channels.
        """
        # Load masks
        if self.rgb:
            if mask == 'weak':
                num = 5
            elif mask == 'strong':
                num = 10
            lum_mask = torch.tensor(np.load(f'{self.root}/rmask_{num}.npy'), dtype=torch.bool)
            chro_cr_mask = torch.tensor(np.load(f'{self.root}/gmask_{num}.npy'), dtype=torch.bool)
            chro_cb_mask = torch.tensor(np.load(f'{self.root}/bmask_{num}.npy'), dtype=torch.bool)            
        else:
            lum_mask = torch.tensor(np.load(f'{self.root}/luminance_mask_{mask}.npy'), dtype=torch.bool)
            chro_cr_mask = torch.tensor(np.load(f'{self.root}/chrominance_cr_mask_{mask}.npy'), dtype=torch.bool)
            chro_cb_mask = torch.tensor(np.load(f'{self.root}/chrominance_cb_mask_{mask}.npy'), dtype=torch.bool)
        self.mask = torch.stack([lum_mask, chro_cr_mask, chro_cb_mask], dim=0).unsqueeze(0).unsqueeze(1).expand(self.batch_size, (self.img_size//self.DCT_MAT_SIZE)**2, 3, self.DCT_MAT_SIZE, self.DCT_MAT_SIZE).reshape(-1, self.DCT_MAT_SIZE, self.DCT_MAT_SIZE).to(self.device)
        
    def create_dct_matrix(self, N):
        """
        Create an NxN DCT transformation matrix using vectorized operations.
        """
        n = torch.arange(N, dtype=torch.float32)  # Ensure float32 for precision
        k = n.view(-1, 1)  # Reshape for broadcasting
        coeffs = torch.sqrt(torch.tensor(2 / N)) * torch.ones(N, dtype=torch.float32)
        coeffs[0] = torch.sqrt(torch.tensor(1 / N))  # First row normalization

        dct_mat = coeffs[:, None] * torch.cos(torch.pi * k * (n + 0.5) / N)
        return dct_mat
        
    def rgb_to_ycrcb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB image (uint8 range [0, 255]) to YCrCb format.
        :param rgb: Input tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.uint8
        :return: Converted tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.float32
        """
        # Ensure input is uint8
        # assert rgb.dtype == torch.uint8, "Input must be in uint8 format (0-255)."
        
        # Convert to float32 for precision
        # rgb = rgb.to(torch.float32)
        
        # RGB channels
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Compute Y, Cr, and Cb
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5

        # Stack to get final tensor
        ycrcb = torch.stack([Y, Cr, Cb], dim=-1)

        return ycrcb
    
    def ycrcb_to_rgb(self, ycrcb: torch.Tensor) -> torch.Tensor:
        """
        Converts a YCrCb image (float32) back to RGB format.
        :param ycrcb: Input tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.float32
        :return: Converted tensor of shape (H, W, 3) or (B, H, W, 3), dtype=torch.uint8
        """
        # Ensure input is float32
        assert ycrcb.dtype == torch.float32, "Input must be in float32 format."

        # Extract Y, Cr, Cb channels
        Y, Cr, Cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
        
        Cr = (Cr - 0.5)
        Cb = (Cb - 0.5)

        # Compute R, G, B
        R = Y + 1.403 * Cr
        G = Y - 0.344 * Cb - 0.714 * Cr
        B = Y + 1.773 * Cb

        # Stack and clip to [0, 255] range
        rgb = torch.stack([R, G, B], dim=-1)

        return rgb
    
    def process_image_torch(self, x0_img, span):
        # Scale to 0-255 and convert to uint8 tensor
        B, _, _, _ = x0_img.shape
        
        if self.rgb:
            ycrcb = x0_img
        else:
            x0_img = (x0_img).clamp(0, 1)
            ycrcb = self.rgb_to_ycrcb(x0_img)

        # Extract channels
        # lum, cr, cb = ycrcb.unbind(dim=-1)

        # Reshape into 8×8 patches
        # patches_lum = lum.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
        # patches_cr = cr.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
        # patches_cb = cb.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(B, -1, 8, 8)
        
        ycrcb = ycrcb.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(-1, 8, 8)
        
        
        # # Apply DCT
        # dct_lum = torch.matmul(self.DCT_MAT, torch.matmul(patches_lum, self.DCT_T))
        # dct_cr = torch.matmul(self.DCT_MAT, torch.matmul(patches_cr, self.DCT_T))
        # dct_cb = torch.matmul(self.DCT_MAT, torch.matmul(patches_cb, self.DCT_T))
        
        ycrcb = torch.bmm(self.DCT_MAT[:B * (self.img_size//self.DCT_MAT_SIZE)**2 * 3], torch.bmm(ycrcb, self.DCT_T[:B * (self.img_size//self.DCT_MAT_SIZE)**2 * 3]))
        ycrcb *= ~self.mask[:B * (self.img_size//self.DCT_MAT_SIZE)**2 * 3]
        
        # Apply masks (ensure they are on the same device)
        # dct_lum *= ~self.lum_mask
        # dct_cr *= ~self.chro_cr_mask
        # dct_cb *= ~self.chro_cb_mask

        # Inverse DCT
        # recon_lum = torch.matmul(self.DCT_T, torch.matmul(dct_lum, self.DCT_MAT))
        # recon_cr = torch.matmul(self.DCT_T, torch.matmul(dct_cr, self.DCT_MAT))
        # recon_cb = torch.matmul(self.DCT_T, torch.matmul(dct_cb, self.DCT_MAT))
        ycrcb = torch.bmm(self.DCT_T[:B * (self.img_size//self.DCT_MAT_SIZE)**2 * 3], torch.bmm(ycrcb, self.DCT_MAT[:B * (self.img_size//self.DCT_MAT_SIZE)**2 * 3]))

        # Reconstruct from patches
        # reconstructed_lum = recon_lum.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
        # reconstructed_cr = recon_cr.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
        # reconstructed_cb = recon_cb.view(B, 4, 4, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, 32, 32)
        ycrcb = ycrcb.view(B, 4, 4, 3, 8, 8).permute(0, 1, 4, 2, 5, 3).reshape(B, 32, 32, 3)

        # Stack channels back together
        # reconstructed_ycrcb = torch.stack([reconstructed_lum, reconstructed_cr, reconstructed_cb], dim=-1)

        # Convert back to RGB
        # reconstructed_rgb = self.ycrcb_to_rgb(reconstructed_ycrcb)
        if self.rgb:
            recon_img = ycrcb
        else:
            recon_img = self.ycrcb_to_rgb(ycrcb)

        # Normalize to -1 to 1 range
        return ((recon_img - 0.5) * span * 2).to(self.device)
    
    def process_image_torch2(self, x0_img):
        B = x0_img.shape[0]
        x0_img = (x0_img).clamp(0, 1)
        ycrcb = self.rgb_to_ycrcb(x0_img)
        
        ycrcb = ycrcb.unfold(1, 8, 8).unfold(2, 8, 8).contiguous().view(-1, 8, 8)
    
    def sample(self, num_samples):
        """
        Sample a batch of images.
        """
        # Sample random indices
        x = torch.randn(num_samples, self.img_size, self.img_size, 3, device=self.device, dtype=self.dtype)
        x = x / (self.span * 2) + 0.5
        x = self.process_image_torch(x, self.span)
        x = x.permute(0, 3, 1, 2)
        if self.mu is not None:
            x = (x - self.mu) / self.std
        return x