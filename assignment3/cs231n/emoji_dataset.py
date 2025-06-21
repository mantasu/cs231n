import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import joblib


import clip
from tqdm.auto import tqdm


def get_text_augs():
    fpath = os.path.join(os.path.dirname(__file__), "datasets/paraphrases_dict.pkl")
    paraphrases_dict = joblib.load(fpath)
    augs = {}
    for text, paraphrases in tqdm(paraphrases_dict.items()):
        paraphrases = paraphrases.split(",")
        text_augs = []
        for p in paraphrases:
            p = p.strip()
            if "\n" in p:
                p = p.split("\n")[-1]
            text_augs.append(p)
        augs[text] = text_augs
    return augs


class ClipEmbed:
    def __init__(self, device):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model = self.model.eval()
        self.device = device

    def embed(self, text):
        with torch.inference_mode():
            text = clip.tokenize(text).to(self.device)
            text_emb = self.model.encode_text(text)[0].cpu()
        return text_emb


class TextEmbedder:
    def __init__(self):
        self.loaded = None

    def load_processed(self, data_path):
        self.loaded = torch.load(data_path)

    def save_processed(self, all_texts, path):
        assert not os.path.exists(path)
        text_embedder = ClipEmbed(device="cuda")
        all_texts = list(set(all_texts))

        # encode all
        idx_mapping = {}
        text_embeddings = []
        for i, text in tqdm(enumerate(all_texts)):
            idx_mapping[text] = i
            text_embeddings.append(text_embedder.embed(text))
        text_embeddings = torch.stack(text_embeddings)

        # PCA
        data = text_embeddings.float().numpy()
        mean = np.mean(data, axis=0)  # Compute mean vector
        centered_data = data - mean
        U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
        components = Vt  # Store all components
        components = torch.from_numpy(components).float()
        mean = torch.from_numpy(mean).float()

        # save
        torch.save(
            {
                "idx_mapping": idx_mapping,
                "embs": text_embeddings,
                "pca_components": components,
                "mean": mean,
            },
            path,
        )

    def embed(self, *, text=None, emb=None, num_pca=None):

        assert (text is None) ^ (emb is None)

        if emb is None:
            emb_idx = self.loaded["idx_mapping"][text]
            emb = self.loaded["embs"][emb_idx].float()

        if num_pca is not None:
            emb = self.encode_pca(emb, num_pca)

        return emb

    def encode_pca(self, emb, num_pca):
        emb = emb - self.loaded["mean"]
        emb = self.loaded["pca_components"][:num_pca] @ emb
        return emb

    def decode_pca(self, emb):
        num_pca = emb.shape[0]
        emb = self.loaded["pca_components"][:num_pca].T @ emb
        emb = emb + self.loaded["mean"]
        return emb


class EmojiDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_path="data/emoji_data.npz",
        text_emb_path="data/text_embeddings.pt",
        num_text_emb_pca=None,
    ):
        
        data_path = os.path.join(os.path.dirname(__file__), "datasets/emoji_data.npz")
        text_emb_path = os.path.join(os.path.dirname(__file__), "datasets/text_embeddings.pt")

        self.load_augs = False
        if self.load_augs:
            print("LOADING AUGS")
            self.text_augs = get_text_augs()
            text_emb_path = "data/text_embeddings_augs.pt"

        loaded = np.load(data_path, allow_pickle=True)
        self.data = [loaded[key].item() for key in loaded]

        if self.load_augs:
            for d in self.data:
                texts = []
                for t in d["texts"]:
                    texts.extend(self.text_augs[t])
                texts = d["texts"] + texts
                d["texts"] = texts

        self.transform = T.Compose(
            [T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]
        )
        self.num_text_emb_pca = num_text_emb_pca
        self.text_embedder = TextEmbedder()
        self.text_embedder.load_processed(text_emb_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgs = self.data[idx]["images"]
        texts = self.data[idx]["texts"]

        # select random image from available images
        img_idx = np.random.choice(len(imgs))
        img = imgs[img_idx]

        # preprocess image
        img = Image.fromarray(img)
        img = self.transform(img)

        # select random text
        text = np.random.choice(texts)
        text_emb = self.text_embedder.embed(text=text, num_pca=self.num_text_emb_pca)
        model_kwargs = {"text_emb": text_emb, "text": text}
        return img, model_kwargs

    def random_model_kwargs(self, n):

        # return n random samples
        idxs = np.random.choice(len(self), n)
        samples = [self.__getitem__(idx) for idx in idxs]
        imgs, model_kwargs = torch.utils.data.default_collate(samples)

        return model_kwargs

    def embed_new_text(self, text, clip_embed):
        text_emb = clip_embed.embed(text).float().cpu()
        if self.num_text_emb_pca is not None:
            text_emb = self.text_embedder.encode_pca(text_emb, self.num_text_emb_pca)
        return text_emb
