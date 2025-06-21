import streamlit as st
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from generator import Generator
import matplotlib.pyplot as plt

device = torch.device("cpu")
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

st.title("Handwritten Image Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate"):
    z = torch.randn(5, 100, device=device)
    labels = torch.full((5,), digit, dtype=torch.long, device=device)

    with torch.no_grad():
        generated_imgs = generator(z)

    grid_img = make_grid(generated_imgs, nrow=5, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(grid_img.permute(1, 2, 0).cpu())
    ax.axis("off")
    st.pyplot(fig)

