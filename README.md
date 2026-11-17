# TikTok Shop AI Studio - Módulo de Inteligência de Concorrentes

Sistema completo para mapeamento de concorrentes, engenharia reversa de criativos virais, benchmark de alto desempenho e ideação estratégica de novos ângulos para o TikTok Shop.

---

## ⚡ Como Iniciar

### 1. Início Rápido (Windows)
Basta clicar duas vezes no arquivo:
```cmd
start.bat
```
Ou pelo terminal:
```powershell
.\venv\Scripts\Activate.ps1
python run.py
```

Acesse no navegador:
* **Painel Web:** [http://localhost:8000](http://localhost:8000)
* **Documentação Interativa (Swagger API):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎯 Funcionalidades Implementadas

### 1. Cadastro e Organização de Perfis Concorrentes
- Cadastro de perfis do TikTok (`@usuario`), apelido, link direto, nicho e notas estratégicas.
- Detecção automática de perfil ao cadastrar links de vídeos.

### 2. Cadastro Híbrido de Vídeos (oEmbed Oficial + Ajuste Manual)
- **Auto-Preenchimento Gratuito:** O sistema consulta o endpoint oficial e gratuito de `oEmbed` do TikTok para obter título/legenda, thumbnail e autor instantaneamente.
- **Entrada Manual Soberana:** Campos para preenchimento de **Visualizações, Curtidas, Comentários e Compartilhamentos**.
- **Respeito aos Dados:** Nunca inventa números; dados inexistentes permanecem como `N/A`.

### 3. Raio-X e Análise Cirúrgica por IA (7 Pilares Persuasivos)
Disseca os vídeos com foco no algoritmo do TikTok Shop:
1. **Gancho Inicial (0-3s)** + Score de retenção.
2. **Headline / Promessa Central**.
3. **Dor ou Desejo Explorado**.
4. **Argumento de Venda & Prova**.
5. **Estrutura Narrativa** (AIDA, PAS, Unboxing com Reação, Antes/Depois).
6. **Demonstração do Produto** (Uso real, close-up, teste extremo).
7. **Chamada para Ação (CTA)** (Carrinho amarelo, bio, engajamento).
- Extrai lições práticas do porquê o vídeo converte.

### 4. Benchmark & Identificação de Padrões Vencedores
- Agrupa múltiplos vídeos do mesmo produto.
- Calcula taxas reais de engajamento: `(Likes + Comentários + Compartilhamentos) / Views`.
- Mapeia automaticamente:
  - Ganchos vencedores comuns dos vídeos com melhor retenção.
  - Estruturas narrativas dominantes.
  - Argumentos de venda que mais geraram atenção e compartilhamento.

### 5. Geração de Ideias Originais Sem Cópia
- A IA sintetiza os padrões vencedores e gera novos ângulos criativos com:
  - Sugestão de Gancho para os primeiros 3 segundos.
  - Estrutura proposta para o roteiro.
  - **Diferenciação Anti-Cópia:** Explicação clara do porquê o novo ângulo é único e superior aos concorrentes.
  - Esboço do roteiro para produção.

### 6. Histórico e Acompanhamento de Evolução
- Permite adicionar novas leituras de métricas ao longo do tempo (snapshots).
- Gráfico interativo com a curva de crescimento de visualizações e curtidas para acompanhar vídeos que viralizam tardiamente.

### 7. Central de Prompts Customizáveis (Prompts Plugáveis)
- Aba exclusiva no painel web para editar os templates de prompts.
- Permite que você aproveite e injete seus modelos proprietários existentes a qualquer momento, sem alterar uma única linha de código.

---

## 🔑 Configuração de IA (Opcional)

O sistema funciona imediatamente em modo local/demonstração. Para ativar as análises profundas em tempo real via Gemini Flash:
1. Abra o arquivo `.env`.
2. Adicione sua chave da API do Google Gemini:
   ```env
   GEMINI_API_KEY=sua_chave_aqui
   GEMINI_MODEL=gemini-2.5-flash
   ```
