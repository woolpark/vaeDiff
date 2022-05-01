import pandas as pd
from PIL import Image
import torchvision.transforms as T

from .data import VAEDataLoader
from .model import PyTorchVAEModels
from .vae_metrics import metric_batch

# Set number of iterations to run
ITERATIONS = 5

def main():
    models = PyTorchVAEModels()
    data = VAEDataLoader()

    test_metrics = {'dvae': [], 'openai': [], 'vqgan': []}
    test_output = {'dvae': [], 'openai': [], 'vqgan': [], 'original': []}

    PIL_transform = T.ToPILImage(mode='RGB')

    early_stop = ITERATIONS

    for idx, (images, _) in enumerate(data.dl):
        if idx >= early_stop:
            break

        for img in images:
            test_output['original'].append(PIL_transform(img))
        images = images.cuda()
        # OpenAI VAE
        enc = models.open_vae.get_codebook_indices(images)
        dec = models.open_vae.decode(enc)
        test_metrics['openai'] = test_metrics['openai'] + metric_batch(images, dec)
        for img in dec:
            test_output['openai'].append(PIL_transform(img))

        # VQGAN VAE
        enc = models.vqgan_vae.model.encode(images)
        dec = models.vqgan_vae.model.decode(enc[0])
        test_metrics['vqgan'] = test_metrics['vqgan'] + metric_batch(images, dec)
        for img in dec:
            test_output['vqgan'].append(PIL_transform(img))

        # Trained DiscVAE
        out = models.model(images, return_recons = True)
        test_metrics['dvae'] = test_metrics['dvae'] + metric_batch(images, out)
        for img in out:
            test_output['dvae'].append(PIL_transform(img))

    for vae in test_metrics.keys():
        metric_array = test_metrics[vae]
    metric_key = metric_array[0].keys()
    convertable_data = {metric: [] for metric in metric_key}
    for metric_item in metric_array:
        for k in metric_key:
            convertable_data[k].append(metric_item[k])
    data = pd.DataFrame.from_dict(convertable_data)
    path = f"./{vae}.csv"
    data.to_csv(path)

    for i in range(len(test_output['original'])):
        path = f"./output/{i}.jpg"
        concat_image = Image.new('RGB', (512, 128))
        concat_image.paste(test_output['original'][i], (0,0))
        concat_image.paste(test_output['dvae'][i], (128,0))
        concat_image.paste(test_output['openai'][i], (256,0))
        concat_image.paste(test_output['vqgan'][i], (384,0))
        concat_image.save(path)

if __name__ == "__main__":
    main()
