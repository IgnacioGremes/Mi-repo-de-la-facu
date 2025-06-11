import numpy as np

def mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def psnr(original, reconstructed):
    mse_val = mse(original, reconstructed)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse_val)) 

def ssim_single(img1, img2):
    mu_x = img1.mean()
    mu_y = img2.mean()
    sigma_x = img1.var()
    sigma_y = img2.var()
    sigma_xy = ((img1 - mu_x) * (img2 - mu_y)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim

def ssim_batch(originals, reconstructions):
    scores = [ssim_single(o.reshape(28, 28), r.reshape(28, 28)) for o, r in zip(originals, reconstructions)]
    return np.mean(scores)

def print_per_image_metrics(originals, pca_recons, vae_recons):
    print("\n=== Métricas por imagen (Top 10) ===")
    print(f"{'Img':<5} {'MSE_PCA':>10} {'MSE_VAE':>10} {'PSNR_PCA':>10} {'PSNR_VAE':>10} {'SSIM_PCA':>10} {'SSIM_VAE':>10}")
    for i in range(len(originals)):
        orig = originals[i]
        pca_rec = pca_recons[i]
        vae_rec = vae_recons[i]
        mse_pca = mse(orig, pca_rec)
        mse_vae = mse(orig, vae_rec)
        psnr_pca = psnr(orig, pca_rec)
        psnr_vae = psnr(orig, vae_rec)
        ssim_pca = ssim_single(orig.reshape(28, 28), pca_rec.reshape(28, 28))
        ssim_vae = ssim_single(orig.reshape(28, 28), vae_rec.reshape(28, 28))
        print(f"{i:<5} {mse_pca:10.5f} {mse_vae:10.5f} {psnr_pca:10.2f} {psnr_vae:10.2f} {ssim_pca:10.5f} {ssim_vae:10.5f}")