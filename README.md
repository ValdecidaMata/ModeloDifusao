# 🎨 Gerador de Dígitos MNIST com Modelo de Difusão

Aplicação Streamlit para treinar e gerar dígitos manuscritos usando um modelo de difusão simplificado.

## 📋 Requisitos

- Python 3.8+
- pip

## 🚀 Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## ▶️ Como Executar

Execute o aplicativo Streamlit:
```bash
streamlit run app_difusao_mnist.py
```

O aplicativo abrirá automaticamente no seu navegador em `http://localhost:8501`

## 📖 Como Usar

### Modo 1: Gerar com modelo não treinado
- Útil para testar a interface
- Gerará apenas ruído aleatório

### Modo 2: Treinar modelo
1. Escolha os hiperparâmetros na barra lateral:
   - Número de épocas (1-10)
   - Batch size (32-256)
   - Learning rate
2. Clique em "Iniciar Treinamento"
3. Aguarde o treinamento completar
4. O modelo será salvo automaticamente como `mini_difusao_mnist.pth`
5. Amostras serão geradas ao final do treinamento

### Modo 3: Carregar checkpoint
1. Faça upload de um arquivo `.pth` previamente treinado
2. Escolha quantas imagens gerar
3. Clique em "Gerar Imagens"

## 🎯 Características

- ✅ Interface web interativa
- ✅ Treinamento com barra de progresso
- ✅ Visualização da curva de loss em tempo real
- ✅ Geração de múltiplas imagens
- ✅ Salvamento e carregamento de checkpoints
- ✅ Suporte para CPU e GPU

## 🔧 Hiperparâmetros do Modelo

- **Timesteps (T):** 200
- **Beta start:** 1e-4
- **Beta end:** 0.02
- **Arquitetura:** MiniUNet com time embedding

## 📝 Notas

- O modelo é simplificado para fins didáticos
- Recomenda-se usar GPU para treinamento mais rápido
- O dataset MNIST será baixado automaticamente na primeira execução
- Para melhores resultados, treine por pelo menos 5 épocas

## 🌐 Deploy no Streamlit Cloud

Para publicar online gratuitamente:

1. Crie uma conta em [streamlit.io/cloud](https://streamlit.io/cloud)
2. Conecte seu repositório GitHub
3. Configure o app apontando para `app_difusao_mnist.py`
4. Deploy!

## 📚 Sobre Modelos de Difusão

Este é um modelo de difusão simplificado baseado em DDPM (Denoising Diffusion Probabilistic Models). O modelo aprende a remover ruído gradualmente através de um processo reverso de difusão.

## 🤝 Contribuições

Sinta-se à vontade para melhorar o código!
