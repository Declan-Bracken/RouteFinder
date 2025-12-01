import torch
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.load_model import load_model, encode_batch, load_images_from_dir

path = "models/pretrained/simclr.ckpt"
image_dir = "src/data/cache"
device = "cpu"
model = load_model(path)
print("Model succesfully loaded")
min_similarity = 0.6

images = load_images_from_dir(image_dir)
embeddings = encode_batch(images, model, device)

# embeddings: (N, D) tensor
norm_embeds = torch.nn.functional.normalize(embeddings, dim=1)

# compute all pairwise cosine similarities
sim_matrix = norm_embeds @ norm_embeds.T
max_pairs_per_page = 2

# Gather all pairs above threshold
pairs_to_plot = []
N = len(images)
for i in range(N):
    for j in range(i + 1, N):
        sim = sim_matrix[i, j].item()
        if sim >= min_similarity:
            pairs_to_plot.append((i, j, sim))

# Paginate
total_pairs = len(pairs_to_plot)
for page_start in range(0, total_pairs, max_pairs_per_page):
    page_pairs = pairs_to_plot[page_start:page_start + max_pairs_per_page]

    fig, axes = plt.subplots(len(page_pairs), 2, figsize=(8, len(page_pairs) * 3))
    if len(page_pairs) == 1:
        axes = [axes]  # make it iterable

    for idx, (i, j, sim) in enumerate(page_pairs):
        axes[idx][0].imshow(images[i])
        axes[idx][0].axis("off")
        axes[idx][0].set_title(f"Image {i}")

        axes[idx][1].imshow(images[j])
        axes[idx][1].axis("off")
        axes[idx][1].set_title(f"Image {j}\nSim={sim:.2f}")

    plt.tight_layout()
    plt.show()

    input(f"Showing pairs {page_start+1}-{page_start+len(page_pairs)} of {total_pairs}. Press Enter for next page...")
# sns.histplot(sims, bins=50)
# plt.xlabel("Cosine similarity")
# plt.ylabel("Frequency")
# plt.title("Distribution of pairwise cosine similarities")
# plt.show()
