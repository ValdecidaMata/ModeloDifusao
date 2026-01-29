# -*- coding: utf-8 -*-
"""
Aplicação Streamlit - Modelo de Difusão para MNIST
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Gerador MNIST - Modelo de Difusão",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Gerador de Dígitos MNIST com Modelo de Difusão")
st.markdown("Modelo baseado em Diffusion Models para gerar dígitos manuscritos")

# ============================================================================
# FUNÇÕES E CLASSES
# ============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def time_embedding(t, dim=64):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    return emb

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(time_embedding(t, 64)).view(-1, 64, 1, 1)
        h = F.relu(self.conv1(x))
        h = h + t_emb[:, :32]
        h = F.relu(self.conv2(h))
        return self.conv3(h)

def forward_diffusion(x0, t, alpha_bar, device):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
    return xt, noise

@torch.no_grad()
def sample(model, n, T, alpha, alpha_bar, beta, device):
    model.eval()
    x = torch.randn(n, 1, 28, 28).to(device)
    
    # Progress bar para o sampling
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(reversed(range(T))):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)
        a = alpha[t]
        a_bar = alpha_bar[t]
        x = (x - (1 - a) / torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a)
        if t > 0:
            x += torch.sqrt(beta[t]) * torch.randn_like(x)
        
        # Atualizar progress bar
        progress = (i + 1) / T
        progress_bar.progress(progress)
        status_text.text(f"Gerando imagens: {int(progress * 100)}%")
    
    progress_bar.empty()
    status_text.empty()
    
    return x

# ============================================================================
# INICIALIZAÇÃO DO MODELO (com cache)
# ============================================================================

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hiperparâmetros
    T = 200
    beta_start = 1e-4
    beta_end = 0.02
    
    beta = torch.linspace(beta_start, beta_end, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Inicializar modelo
    model = MiniUNet().to(device)
    
    return model, device, T, alpha, alpha_bar, beta

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Carregar modelo
model, device, T, alpha, alpha_bar, beta = load_model()

st.sidebar.header("⚙️ Configurações")

# Opção de treinar ou usar modelo pré-treinado
mode = st.sidebar.radio(
    "Modo de operação:",
    ["Gerar com modelo não treinado", "Treinar modelo", "Carregar checkpoint"]
)

# ============================================================================
# MODO: GERAR
# ============================================================================

if mode == "Gerar com modelo não treinado":
    st.info("⚠️ Modelo não treinado. As imagens geradas serão ruído aleatório. Treine o modelo primeiro para resultados melhores.")
    
    num_images = st.sidebar.slider("Número de imagens", 4, 16, 16)
    
    if st.button("🎲 Gerar Imagens", type="primary"):
        with st.spinner("Gerando imagens..."):
            seed_everything(random.randint(0, 10000))
            samples = sample(model, num_images, T, alpha, alpha_bar, beta, device).cpu()
        
        # Visualização
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i in range(num_images):
            axes[i].imshow(samples[i].squeeze(), cmap="gray")
            axes[i].axis("off")
        
        for i in range(num_images, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# MODO: TREINAR
# ============================================================================

elif mode == "Treinar modelo":
    st.subheader("🔧 Treinamento do Modelo")
    
    epochs = st.sidebar.slider("Número de épocas", 1, 10, 3)
    batch_size = st.sidebar.slider("Batch size", 32, 256, 128)
    learning_rate = st.sidebar.select_slider(
        "Learning rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3],
        value=1e-3
    )
    
    if st.button("🚀 Iniciar Treinamento", type="primary"):
        # Preparar dataset
        with st.spinner("Carregando dataset MNIST..."):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            dataset = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transform
            )
            
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Reinicializar modelo e otimizador
        seed_everything(42)
        model = MiniUNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Treinamento
        st.info(f"Treinando por {epochs} épocas...")
        
        loss_history = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (x, _) in enumerate(loader):
                x = x.to(device)
                t = torch.randint(0, T, (x.size(0),), device=device)
                
                xt, noise = forward_diffusion(x, t, alpha_bar, device)
                noise_pred = model(xt, t)
                
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # Atualizar status a cada 50 batches
                if batch_idx % 50 == 0:
                    progress = (epoch + batch_idx / len(loader)) / epochs
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Época {epoch+1}/{epochs} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}"
                    )
            
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            
            # Atualizar gráfico de loss
            fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
            ax_loss.plot(loss_history, marker='o')
            ax_loss.set_xlabel("Época")
            ax_loss.set_ylabel("Loss (MSE)")
            ax_loss.set_title("Curva de Treinamento")
            ax_loss.grid(True, alpha=0.3)
            loss_chart.pyplot(fig_loss)
            plt.close(fig_loss)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Treinamento concluído! Loss final: {loss_history[-1]:.4f}")
        
        # Salvar checkpoint
        checkpoint = {
            "model_state": model.state_dict(),
            "T": T,
            "beta_start": beta_start,
            "beta_end": beta_end
        }
        
        torch.save(checkpoint, "mini_difusao_mnist.pth")
        st.success("💾 Modelo salvo em: mini_difusao_mnist.pth")
        
        # Gerar amostras
        st.subheader("🎨 Amostras Geradas")
        with st.spinner("Gerando imagens..."):
            samples = sample(model, 16, T, alpha, alpha_bar, beta, device).cpu()
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            axes[i//4, i%4].imshow(samples[i].squeeze(), cmap="gray")
            axes[i//4, i%4].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# MODO: CARREGAR CHECKPOINT
# ============================================================================

elif mode == "Carregar checkpoint":
    st.subheader("📂 Carregar Modelo Treinado")
    
    uploaded_file = st.file_uploader(
        "Faça upload do checkpoint (.pth)",
        type=["pth"]
    )
    
    if uploaded_file is not None:
        try:
            checkpoint = torch.load(uploaded_file, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            st.success("✅ Modelo carregado com sucesso!")
            
            num_images = st.slider("Número de imagens", 4, 16, 16)
            
            if st.button("🎲 Gerar Imagens", type="primary"):
                with st.spinner("Gerando imagens..."):
                    seed_everything(random.randint(0, 10000))
                    samples = sample(model, num_images, T, alpha, alpha_bar, beta, device).cpu()
                
                cols = 4
                rows = (num_images + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
                axes = axes.flatten() if num_images > 1 else [axes]
                
                for i in range(num_images):
                    axes[i].imshow(samples[i].squeeze(), cmap="gray")
                    axes[i].axis("off")
                
                for i in range(num_images, len(axes)):
                    axes[i].axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {str(e)}")

# ============================================================================
# RODAPÉ
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Sobre")
st.sidebar.info(
    "Este é um modelo de difusão simplificado treinado no dataset MNIST. "
    "O modelo aprende a remover ruído gradualmente para gerar dígitos manuscritos."
)

st.sidebar.markdown(f"**Device:** {device}")
st.sidebar.markdown(f"**Timesteps:** {T}")
