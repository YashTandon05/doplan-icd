# doPlan: Instruction Composition & Decomposition (ICD)

Testing whether driving intent survives the round trip:
`L → G(L) → F(G(L))`

## Setup

```bash
git clone https://github.com/YashTandon05/doplan-icd.git
cd doplan-icd
pip install -r requirements.txt

# install Ollama from https://ollama.com, then:
ollama pull qwen2.5vl:7b
ollama run qwen2.5vl:7b   # confirm it responds
```
