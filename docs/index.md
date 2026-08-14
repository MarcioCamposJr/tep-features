# tepfeatures

Bem-vindo à documentação da biblioteca **tepfeatures**! 🧠

Esta biblioteca Python foi desenhada para realizar a extração automatizada de features avançadas de **TMS-Evoked Potentials (TEPs)**. Focada na manipulação de objetos `mne.Evoked`, ela estende o ecosistema MNE-Python entregando descritores de morfologia, features espaciais e dinâmicas de microestados de forma estruturada.

## Visão Geral

- **Core**: O `TEPExtractor` coordena a extração a partir de janelas temporais dinâmicas e validadas (ex: N15, P30, etc).
- **Features Temporais**: Extração inteligente baseada na polaridade do pico (amplitude, latência) e descritores morfológicos entre picos (slope, peak-to-peak amplitude).
- **Features Espaciais**: Global e Local Mean Field Power, captura automática de picos da GFP e matriz de Similaridade Topográfica por co-seno.
- **Features Dinâmicas**: Microestados baseados num algoritmo K-Means modificado.
