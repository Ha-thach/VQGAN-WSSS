import yaml
import torch
from omegaconf import OmegaConf 
from taming.models.vqgan import VQModel, GumbelVQ

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config))) #.dump to print the config format nicely if display=True
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False): # load a VQGAN model from a config and checkpoint
  if is_gumbel: # use GumbelVQ 
    model = GumbelVQ(**config.model.params)
  else: # use VQModel 
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x): # convert from range [0, 1] to [-1, 1] to use tanh() in output decoder
  x = 2.*x - 1.
  return x

def custom_to_pil(x): # convert a tensor to a PIL Image
  x = x.detach().cpu() # .detach to create a tensor that shares the same dataa as the original one but does not require gradients
  x = torch.clamp(x, -1., 1.) # .clamp to limit the values to a given range 
  x = (x + 1.)/2. # scale to [0, 1]
  x = x.permute(1,2,0).numpy() # convert to HWC
  x = (255*x).astype(np.uint8) # scale to [0, 255]
  x = Image.fromarray(x) # convert to PIL Image
  if not x.mode == "RGB": # ensure 3 channels
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec
def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    #if s < target_image_size:
     #   raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img


def reconstruction_pipeline_from_path(image_path, size=384):
    # Load local image
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess to VQGAN style tensor
    x_vqgan = preprocess(img, target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(DEVICE)

    print(f"✅ Input image loaded: {image_path}")
    print(f"📐 Tensor shape: {x_vqgan.shape}")

    # Reconstructions
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)     # f8, 8192
    x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model16384)     # f16, 16384
    x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024)      # f16, 1024
    x3 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_bcss)     # f8, 1024 (your retrain)

    # Convert outputs to PIL
    img = stack_reconstructions(
        custom_to_pil(preprocess_vqgan(x_vqgan[0])),
        custom_to_pil(x0[0]),
        custom_to_pil(x1[0]),
        custom_to_pil(x2[0]),
        custom_to_pil(x3[0]),
        titles=titles
    )

    return img

def main(config):
    config_model = load_config(config.config_nam, display=False)
    model = load_vqgan(config_model, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)
    x_vqgan = preprocess(image, target_image_size=size, map_dalle=False)
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
    titles = [
    "Input",
    "VQGAN (f8, 8192)",
    "VQGAN (f16, 16384)",
    "VQGAN (f16, 1024)",
    "BCSS-retrain (f8, 1024)"
]