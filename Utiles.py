import torch
import torch.nn.functional as F


def psnr(x, y_out):
    mse = torch.mean((x - y_out) ** 2)
    max_val = 1.0  # 图像像素值范围默认为[0, 1]，如果不是，请根据实际情况调整
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr

def ssim(x,y_out):
    max_val = tf.math.reduce_max(x)
    ssim_val = tf.image.ssim(x,y_out,max_val=max_val)
    return ssim_val

def simulate_noise(volume, noise_level):
    shape = volume.shape
    noise = torch.normal(mean=0.0, std=noise_level, size=shape)
    noise_simulated_data = volume + noise
    return noise_simulated_data

def add_salt_pepper_noise(x, prob):
    shape = x.shape
    salt_pepper_num = int(prob * shape.numel())
    coords = torch.randint(0, shape[-1], (salt_pepper_num, len(shape)))

    mask = torch.zeros(shape)
    mask[coords] = 1
    mask = mask.type_as(x)

    salt_mask = torch.randint(0, 2, (salt_pepper_num,))
    salt_mask = salt_mask.type_as(x)
    pepper_mask = 1 - salt_mask

    mask = salt_mask * mask + pepper_mask * (1 - mask) * 255
    return x + mask

def add_gaussian_noise(image, var):
    gauss = torch.normal(0, var, size=image.shape)
    noise_img = image + gauss/255
    return noise_img

def loss(x, y_out):
    loss_val = F.mse_loss(y_out, x)
    return loss_val

def MSA(x, y_out):
    mse = torch.mean((y_out - x) ** 2)
    return mse


def random_rotate(img, rot):
    img = torch.tensor(img)
    img = img.permute(2, 0, 1)
    
    img = torch.rot90(img, rot, dims=(1, 2))

    img = img.permute(1, 2, 0)#.numpy()
    return img

def random_zoom(img, scale):
    h, w = img.shape[0], img.shape[1]
    new_h, new_w = int(h * scale), int(w * scale)

    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)
    return img

def train1(image, b, rot, scale):
    K = 16
    H1, W1, B1 = image.shape
    h, w = 100, 100
    top = np.random.randint(0, H1 - h)
    left = np.random.randint(0, W1 - w)
    cropped_image = image[top:top + h, left:left + w]
    cropped_image = random_rotate(image, rot)
    #cropped_image = random_zoom(cropped_image, scale)
    spatial_band = cropped_image[:, :, b].unsqueeze(0)

    if b < K:
        spectral_volume = cropped_image[:, :, :K]
    elif b >= B1 - K // 2:
        spectral_volume = cropped_image[:, :, -K:]
    else:
        start = b - K // 2
        end = b + K // 2
        spectral_volume = cropped_image[:, :, start:end]

    spatial_band = spatial_band.unsqueeze(0)
    spectral_volume = spectral_volume.unsqueeze(0)
    spectral_volume = spectral_volume.permute(0,3,1,2)
    return spatial_band, spectral_volume

def valid1(image, b):
    K = 16
    H, W, B = image.shape

    spatial_band = image[:, :, b].unsqueeze(0)

    if b < K:
        spectral_volume = image[:, :, :K]
    elif b >= B - K // 2:
        spectral_volume = image[:, :, -K:]
    else:
        start = b - K // 2
        end = b + K // 2
        spectral_volume = image[:, :, start:end]

    spatial_band = spatial_band.unsqueeze(0)
    spectral_volume = spectral_volume.unsqueeze(0)
    return spatial_band, spectral_volume